"""Underlying-price first-passage exits for a frozen option-entry rule."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import FEATURE_COLUMNS, StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.reversal_refinement import SIGNAL_COLUMNS
from research_engine.analysis.stock_strategy_search import aggregate, fresh_gate, pick

VARIANTS = {"baseline": (None, None), "take_half": (.5, None), "stop_half": (None, .5),
            "take_half_stop_half": (.5, .5), "take_full_stop_half": (1., .5)}


def underlying_exit_scheduler(frame, take, stop):
    source = frame.sort_values("timestamp").copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.timestamp, utc=True)).as_unit("ns")
    if not source.index.is_unique:
        raise ValueError("unique stock timestamps required")
    valid = (source.qqq_depth_age_seconds.between(0, 1) & source.qqq_bid.gt(0) &
             source.qqq_ask.ge(source.qqq_bid) & np.isfinite(source.qqq_bid) & np.isfinite(source.qqq_ask))
    mids = ((source.qqq_bid + source.qqq_ask) / 2).where(valid)

    def schedule(signal, symbol, entry_time, deadline):
        shock = abs(signal["spot"] - signal["spot"] / (1 + signal["r60"]))
        if not np.isfinite(shock) or shock <= 0:
            raise ValueError("positive observable stock shock required")
        details = dict(exit_reason="time", exit_trigger_time=pd.NaT, exit_trigger_spot=np.nan, stock_shock=shock)
        # A vectorized first crossing is equivalent to checking in timestamp order.
        # Data after the first crossing never sets its timestamp or threshold.
        scan = mids.loc[entry_time + pd.Timedelta(seconds=30): deadline - pd.Timedelta(seconds=1)]
        change = signal["direction"] * (scan - signal["spot"])
        hit_take = change.ge(take * shock) if take is not None else pd.Series(False, index=scan.index)
        hit_stop = change.le(-stop * shock) if stop is not None else pd.Series(False, index=scan.index)
        crossings = scan[hit_take | hit_stop]
        if not crossings.empty:
            when = crossings.index[0]
            details.update(exit_reason="take" if hit_take.loc[when] else "stop",
                           exit_trigger_time=when, exit_trigger_spot=float(scan.loc[when]))
            return when + pd.Timedelta(seconds=1), details
        return deadline, details

    return schedule


def run(data_root, baseline_root, output):
    meta = json.loads((baseline_root / "study.json").read_text())
    if meta["context"] != "morning" or meta["strict_entry"]:
        raise ValueError("requires original morning signals")
    for record in meta["source_manifest"]:
        if file_hash(data_root / record["file"]) != record["sha256"]:
            raise ValueError("source data changed")
    output.mkdir(parents=True, exist_ok=False)
    settings = replace(StudySettings(), holding_seconds=300)
    daily, ledgers = [], []
    for day in meta["discovery_dates"] + meta["check_dates"]:
        source = pd.read_parquet(baseline_root / "opportunities" / (day + ".parquet"))
        signals = source[source.factor.eq("reversal60") & source.horizon.eq(300)][SIGNAL_COLUMNS].copy()
        frame = pd.read_parquet(data_root / "features" / (day + ".parquet"), columns=FEATURE_COLUMNS)
        books = read_books(data_root / "normalized" / day / "events.parquet", day)
        for name, (take, stop) in VARIANTS.items():
            scheduler = None if name == "baseline" else underlying_exit_scheduler(frame, take, stop)
            ledger = simulate_single_day(signals.assign(factor=name), books, settings,
                                         entry_gate=fresh_gate(books), exit_scheduler=scheduler)
            if name == "baseline":
                ledger["exit_reason"] = "time"
            ledger["horizon"] = 300
            metrics = execution_metrics(ledger)
            metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
            daily.append(dict(date=day, factor=name, horizon=300, **metrics))
            ledgers.append(ledger)
        print("Completed " + day, flush=True)
    daily = pd.DataFrame(daily)
    ledger = pd.concat(ledgers, ignore_index=True)
    discovery = aggregate(daily, meta["discovery_dates"])
    selected = pick(discovery)
    checking = aggregate(daily, meta["check_dates"]).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", checking),
                          ("all_dates", aggregate(daily, meta["discovery_dates"] + meta["check_dates"]))]:
        values.to_csv(output / (name + ".csv"), index=False)
    ledger.to_parquet(output / "opportunities.parquet", index=False)
    ledger[ledger.status.eq("closed")].groupby(["date", "factor", "exit_reason"]).agg(
        count=("net_pnl", "size"), mean_net=("net_pnl", "mean"), mean_holding=("holding_seconds", "mean")
    ).to_csv(output / "exit_reasons.csv")
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/adaptive_exit_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    metadata = dict(study="stock_first_passage_option_exits", variants=VARIANTS, settings=asdict(settings),
                    discovery_dates=meta["discovery_dates"], check_dates=meta["check_dates"], baseline_study=baseline_root.name,
                    source_manifest=meta["source_manifest"], baseline_metadata_sha256=file_hash(baseline_root / "study.json"),
                    protocol_sha256=file_hash(protocol),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 正股触发提前平仓", "", "固定入场信号，首次触发后再延迟1秒退出；费用和不可成交状态全部保留。",
        "所有日期均为探索，未重新优化阈值。", "", "## 前段全部结果", "", "```", discovery.to_string(index=False), "```", "",
        "## 前段候选在后段的结果", "", "```", checking.to_string(index=False), "```", ""]), encoding="utf-8")
    print(discovery.to_string(index=False))
    print("Selected check:"); print(checking.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("adaptive_exit_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, args.baseline_root, output)


if __name__ == "__main__":
    main()
