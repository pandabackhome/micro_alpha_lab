"""Frozen morning reversal: latency, quote-age sensitivity and raw fill audits."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import BOOK_COLUMNS, StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.reversal_refinement import SIGNAL_COLUMNS
from research_engine.analysis.stock_strategy_search import aggregate, fresh_gate

VARIANTS = {"baseline": {}, "exit_age1": {"max_quote_age_seconds": 1.},
            "latency2": {"latency_seconds": 2}, "latency5": {"latency_seconds": 5},
            "slippage5bps": {"slippage_bps": 5.}}


def audit_raw_fills(ledger, events, settings):
    """Independent last-update lookup, including invalid updates and duplicate order."""
    source = events.copy()
    source["available_at"] = pd.to_datetime(source.available_at, utc=True).dt.as_unit("ns")
    groups = {symbol: group.sort_values("available_at", kind="stable") for symbol, group in source.groupby("symbol")}
    checked = 0
    for row in ledger[ledger.status.isin(["closed", "unclosed"])].itertuples():
        group = groups[row.symbol]
        sides = [("entry", row.entry_time, row.entry_ask, row.entry_quote_age, 1.)]
        if row.status == "closed":
            sides.append(("exit", row.exit_time, row.exit_bid, row.exit_quote_age, settings.max_quote_age_seconds))
        for side, when, actual_price, actual_age, max_age in sides:
            eligible = group[group.available_at.le(when)]
            if eligible.empty:
                raise ValueError("fill without an available raw quote")
            quote = eligible.iloc[-1]
            age = (when - quote.available_at).total_seconds()
            price = quote.best_ask if side == "entry" else quote.best_bid
            if not (0 <= age <= max_age and quote.best_bid > 0 and quote.best_ask >= quote.best_bid and
                    quote.best_bid_size >= 1 and quote.best_ask_size >= 1 and
                    np.isfinite([quote.best_bid, quote.best_ask, quote.best_bid_size, quote.best_ask_size]).all()):
                raise ValueError("invalid raw fill quote")
            if not np.isclose(price, actual_price, rtol=0, atol=1e-9) or not np.isclose(age, actual_age, rtol=0, atol=1e-6):
                raise ValueError("raw quote price or age mismatch")
            if quote.option_right != row.right or quote.option_strike != row.strike:
                raise ValueError("raw contract mismatch")
            if side == "entry" and quote.available_at <= row.timestamp:
                raise ValueError("entry quote predates signal")
            checked += 1
        slip = settings.slippage_bps / 10000
        debit = row.entry_ask * (1 + slip) * settings.multiplier + settings.commission_per_contract
        if not np.isclose(row.entry_debit, debit, rtol=0, atol=1e-8):
            raise ValueError("entry debit mismatch")
        if row.status == "closed":
            net = row.exit_bid * (1 - slip) * settings.multiplier - settings.commission_per_contract - debit
            if not np.isclose(row.net_pnl, net, rtol=0, atol=1e-8):
                raise ValueError("net cash flow mismatch")
    return checked


def run(data_root, baseline_root, strict_root, output):
    meta = json.loads((baseline_root / "study.json").read_text())
    if meta["context"] != "morning" or meta["strict_entry"]:
        raise ValueError("requires original morning signal study")
    for record in meta["source_manifest"]:
        if file_hash(data_root / record["file"]) != record["sha256"]:
            raise ValueError("source data changed")
    output.mkdir(parents=True, exist_ok=False)
    daily, ledgers, audits = [], [], []
    for day in meta["discovery_dates"] + meta["check_dates"]:
        source = pd.read_parquet(baseline_root / "opportunities" / (day + ".parquet"))
        signals = source[source.factor.eq("reversal60") & source.horizon.eq(300)][SIGNAL_COLUMNS].copy()
        path = data_root / "normalized" / day / "events.parquet"
        books = read_books(path, day)
        raw = pd.read_parquet(path, columns=BOOK_COLUMNS,
                              filters=[("kind", "=", "depth"), ("symbol", "in", list(books.books))])
        for name, changes in VARIANTS.items():
            settings = replace(StudySettings(), holding_seconds=300, **changes)
            prepared = signals.assign(factor=name)
            ledger = simulate_single_day(prepared, books, settings, entry_gate=fresh_gate(books))
            ledger["horizon"] = 300
            checked = audit_raw_fills(ledger, raw, settings)
            if name == "baseline":
                previous = pd.read_parquet(strict_root / "opportunities" / (day + ".parquet"))
                previous = previous[previous.factor.eq("reversal60") & previous.horizon.eq(300)]
                keys = ["timestamp", "status", "symbol", "entry_time", "exit_time", "net_pnl"]
                pd.testing.assert_frame_equal(ledger[keys].reset_index(drop=True), previous[keys].reset_index(drop=True))
            metrics = execution_metrics(ledger)
            metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
            daily.append(dict(date=day, factor=name, horizon=300, **metrics))
            audits.append(dict(date=day, variant=name, checked_quotes=checked, baseline_matched=name == "baseline"))
            ledgers.append(ledger)
        print("Audited " + day, flush=True)
    daily = pd.DataFrame(daily)
    summary = pd.concat([aggregate(daily, dates).assign(part=part) for part, dates in
                         [("discovery", meta["discovery_dates"]), ("check", meta["check_dates"]),
                          ("all", meta["discovery_dates"] + meta["check_dates"])]], ignore_index=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    closed = ledger[ledger.factor.eq("baseline") & ledger.status.eq("closed")]
    concentration = dict(closed=len(closed), total_net=float(closed.net_pnl.sum()),
                         median_net=float(closed.net_pnl.median()), win_rate=float(closed.net_pnl.gt(0).mean()),
                         mean_entry_debit=float(closed.entry_debit.mean()), max_entry_debit=float(closed.entry_debit.max()),
                         without_best_trade=float(closed.net_pnl.sum() - closed.net_pnl.max()),
                         without_best_five_trades=float(closed.net_pnl.sum() - closed.net_pnl.nlargest(5).sum()),
                         raw_quotes_checked=int(pd.DataFrame(audits).checked_quotes.sum()))
    daily.to_csv(output / "daily.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "raw_quote_audit.csv", index=False)
    ledger.to_parquet(output / "opportunities.parquet", index=False)
    (output / "concentration.json").write_text(json.dumps(concentration, indent=2) + "\n", encoding="utf-8")
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/reversal_execution_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    metadata = dict(study="frozen_reversal_execution_audit", variants=VARIANTS, settings=asdict(replace(StudySettings(), holding_seconds=300)),
                    discovery_dates=meta["discovery_dates"], check_dates=meta["check_dates"], baseline_study=baseline_root.name,
                    strict_study=strict_root.name, source_manifest=meta["source_manifest"], protocol_sha256=file_hash(protocol),
                    baseline_metadata_sha256=file_hash(baseline_root / "study.json"),
                    strict_metadata_sha256=file_hash(strict_root / "study.json"),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 固定早盘反转执行复核", "", "未重新筛选规则，各变体独立重放。金额为每张美元，净收益已扣费用；所有日期均为探索。",
        "", "```", summary.to_string(index=False), "```", "", "## 基准集中度与原始报价核对", "", "```json",
        json.dumps(concentration, indent=2), "```", ""]), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(concentration))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("reversal_execution_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, args.baseline_root, args.strict_root, output)


if __name__ == "__main__":
    main()
