"""Causal multi-position accounting for unchanged underlying entry signals."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.reversal_refinement import SIGNAL_COLUMNS
from research_engine.analysis.stock_strategy_search import aggregate, fresh_gate, pick

CAPITAL = 1000.
CAPACITIES = (1, 2, 4)


def simulate_capacity(signals, books, settings, capacity, capital=CAPITAL):
    if capacity < 1 or capital <= 0:
        raise ValueError("positive capacity and capital required")
    rows, active, cash, peak_debit, peak_positions = [], [], capital, 0., 0
    halted = False
    def release(when):
        nonlocal active, cash, halted
        remaining = []
        for position in active:
            if position["status"] == "closed" and position["exit_time"] <= when:
                cash += position["entry_debit"] + position["net_pnl"]
            else:
                remaining.append(position)
                if position["status"] == "unclosed" and position["last_exit_check"] <= when:
                    halted = True
        active = remaining
    for signal in signals[signals.scheduled].sort_values("timestamp").to_dict("records"):
        when = signal["timestamp"]
        release(when)
        row = dict(signal, horizon=settings.holding_seconds, status=signal["signal_status"], reason="", policy_eligible=False)
        if signal["signal_status"] != "signal":
            rows.append(row)
            continue
        row["policy_eligible"] = True
        if halted or len(active) >= capacity:
            row.update(status="busy", reason="known unresolved exit" if halted else "position capacity reached")
            rows.append(row)
            continue
        one = simulate_single_day(pd.DataFrame([signal]), books, settings, entry_gate=fresh_gate(books)).iloc[0].to_dict()
        one["horizon"] = settings.holding_seconds
        if one["status"] in ("closed", "unclosed"):
            release(one["entry_time"])
            if halted or one["entry_debit"] > cash:
                row.update(status="cash_rejected", reason="known unresolved exit" if halted else "insufficient current cash")
                rows.append(row)
                continue
            one["cash_before_entry"] = cash
            cash -= one["entry_debit"]
            one["cash_after_entry"] = cash
            active.append(one)
            peak_debit = max(peak_debit, sum(position["entry_debit"] for position in active))
            peak_positions = max(peak_positions, len(active))
        rows.append(one)
    if len(signals):
        release(signals.session_close.max())
    ledger = pd.DataFrame(rows)
    stats = execution_metrics(ledger)
    stats.update(entry_rejected=int(ledger.status.eq("entry_rejected").sum()),
                 cash_rejected=int(ledger.status.eq("cash_rejected").sum()), initial_capital=capital,
                 end_cash=cash, max_open_debit=peak_debit, max_positions=peak_positions,
                 stress_return=stats["net_with_unclosed_zero_recovery"] / capital)
    if abs(cash - capital - stats["net_with_unclosed_zero_recovery"]) > 1e-7:
        raise ValueError("cash conservation failed")
    return ledger, stats


def run(archive_data, local_data, phase_root, new_root, output):
    phase_meta = json.loads((phase_root / "study.json").read_text())
    phase = pd.read_parquet(phase_root / "opportunities.parquet")
    phase = phase[phase.exit_style.eq("baseline") & phase.latency.eq(1)].copy()
    phase["sample"] = "old"
    new = pd.read_parquet(new_root / "opportunities.parquet")
    new = new[new.factor.eq("clock_reversal")].copy()
    new["phase"], new["sample"] = 30, "added"
    inputs = pd.concat([phase, new], ignore_index=True)
    settings = replace(StudySettings(), holding_seconds=300)
    output.mkdir(parents=True, exist_ok=False)
    daily, ledgers, manifests = [], [], []
    for day, group in inputs.groupby("date"):
        origin = "local" if group["sample"].eq("added").any() else "archive"
        root = local_data if origin == "local" else archive_data
        source = root / "normalized" / day / "events.parquet"
        books = read_books(source, day)
        manifests.append(dict(origin=origin, file="normalized/" + day + "/events.parquet", sha256=file_hash(source)))
        for phase_value, panel in group.groupby("phase"):
            signals = panel[SIGNAL_COLUMNS].copy()
            for capacity in CAPACITIES:
                factor = "phase" + str(phase_value) + "_cap" + str(capacity)
                ledger, stats = simulate_capacity(signals.assign(factor=factor), books, settings, capacity)
                ledger["phase"], ledger["capacity"] = phase_value, capacity
                if capacity == 1:
                    keys = ["timestamp", "status", "symbol", "entry_time", "exit_time", "net_pnl"]
                    pd.testing.assert_frame_equal(ledger[keys].reset_index(drop=True), panel[keys].reset_index(drop=True))
                daily.append(dict(date=day, factor=factor, horizon=300, phase=phase_value, capacity=capacity, **stats))
                ledgers.append(ledger)
        print("Accounted positions " + day, flush=True)
    daily = pd.DataFrame(daily)
    discovery = aggregate(daily, phase_meta["discovery_dates"])
    selected = pick(discovery[discovery.factor.str.startswith("phase30_")])
    check = aggregate(daily, phase_meta["check_dates"]).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    added = aggregate(daily, sorted(new.date.unique())).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", check),
                          ("added_dates_check", added), ("old_all_dates", aggregate(daily, phase_meta["discovery_dates"] + phase_meta["check_dates"]))]:
        values.to_csv(output / (name + ".csv"), index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(output / "opportunities.parquet", index=False)
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/position_capacity_protocol.md"
    (output / "protocol.md").write_bytes(protocol.read_bytes())
    metadata = dict(study="stock_signal_position_capacity", capacities=CAPACITIES, daily_initial_capital=CAPITAL,
                    settings=asdict(settings), discovery_dates=phase_meta["discovery_dates"], check_dates=phase_meta["check_dates"],
                    added_dates=sorted(new.date.unique()), source_manifest=manifests, protocol_sha256=file_hash(protocol),
                    phase_study=phase_root.name, phase_metadata_sha256=file_hash(phase_root / "study.json"),
                    new_study=new_root.name, new_metadata_sha256=file_hash(new_root / "study.json"),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 不同持仓上限，统一每日1000美元初始资金", "", "入场规则未改；未平仓零回收压力损益与现金守恒核对，资金只在实际退出后释放。",
        "旧四相位全部报告，候选只能从原第30秒容量方案选择。新增日期已被读取，不再作为新留出。", "",
        "## 前段全部结果", "", "```", discovery.to_string(index=False), "```", "",
        "## 选出方案旧后段", "", "```", check.to_string(index=False), "```", "",
        "## 选出方案新增日期", "", "```", added.to_string(index=False), "```", "",
        "资金与风险占用见 daily.csv 的 initial_capital、max_open_debit、max_positions、stress_return。不得直接把更高容量的收益视为更高效率。", ""]), encoding="utf-8")
    print(discovery.to_string(index=False))
    print("Old check:"); print(check.to_string(index=False))
    print("Added check:"); print(added.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data", type=Path, required=True)
    parser.add_argument("--local-data", type=Path, default=Path("data"))
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("position_capacity_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data, args.local_data, args.phase_root, args.new_root, output)


if __name__ == "__main__":
    main()
