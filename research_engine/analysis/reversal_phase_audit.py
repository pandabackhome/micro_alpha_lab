"""Audit minute sampling phase without changing stock windows or thresholds."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_engine.analysis.adaptive_exit_search import underlying_exit_scheduler
from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.stock_strategy_search import COLUMNS, aggregate, calibrate, fresh_gate, signals_for, stock_observations

PHASES = (0, 15, 30, 45)
LATENCIES = (1, 5)


def run(data_root, adaptive_root, output):
    settings = replace(StudySettings(), holding_seconds=300)
    meta = json.loads((adaptive_root / "study.json").read_text())
    for item in meta["source_manifest"]:
        if file_hash(data_root / item["file"]) != item["sha256"]:
            raise ValueError("source data changed")
    previous = pd.read_parquet(adaptive_root / "opportunities.parquet")
    output.mkdir(parents=True, exist_ok=False)
    history, daily, ledgers = [], [], []
    for path in sorted((data_root / "features").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=COLUMNS)
        ordinary = stock_observations(frame, settings)
        current = stock_observations(frame, settings, sampling_seconds=15)
        calibrated = calibrate(current, history, settings)
        history.append(ordinary)
        if path.stem not in meta["discovery_dates"] + meta["check_dates"]:
            continue
        signals = signals_for(calibrated, "morning")
        signals = signals[signals.factor.eq("reversal60")].copy()
        books = read_books(data_root / "normalized" / path.stem / "events.parquet", path.stem)
        for phase in PHASES:
            prepared = signals.copy()
            prepared["scheduled"] = prepared.seconds_from_open.mod(60).eq(phase)
            for exit_style in ("baseline", "stop_half"):
                scheduler = None if exit_style == "baseline" else underlying_exit_scheduler(frame, None, .5)
                for latency in LATENCIES:
                    factor = exit_style + "_phase" + str(phase) + "_lag" + str(latency)
                    execution = replace(settings, latency_seconds=latency)
                    ledger = simulate_single_day(prepared.assign(factor=factor), books, execution,
                                                 entry_gate=fresh_gate(books), exit_scheduler=scheduler)
                    ledger["horizon"] = 300
                    ledger["phase"], ledger["latency"], ledger["exit_style"] = phase, latency, exit_style
                    if phase == 30 and latency == 1:
                        old = previous[previous.date.eq(path.stem) & previous.factor.eq(exit_style)]
                        keys = ["timestamp", "status", "symbol", "entry_time", "exit_time", "net_pnl"]
                        pd.testing.assert_frame_equal(ledger[keys].reset_index(drop=True), old[keys].reset_index(drop=True))
                    metrics = execution_metrics(ledger)
                    metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
                    daily.append(dict(date=path.stem, factor=factor, horizon=300, phase=phase, latency=latency,
                                      exit_style=exit_style, **metrics))
                    ledgers.append(ledger)
        print("Completed " + path.stem, flush=True)
    daily = pd.DataFrame(daily)
    summary = pd.concat([aggregate(daily, dates).assign(part=part) for part, dates in
                         [("discovery", meta["discovery_dates"]), ("check", meta["check_dates"]),
                          ("all", meta["discovery_dates"] + meta["check_dates"])]], ignore_index=True)
    summary = summary.merge(daily[["factor", "phase", "latency", "exit_style"]].drop_duplicates(), on="factor", validate="many_to_one")
    daily.to_csv(output / "daily.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(output / "opportunities.parquet", index=False)
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/reversal_phase_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    metadata = dict(study="frozen_reversal_sampling_phase", phases=PHASES, latencies=LATENCIES, settings=asdict(settings),
                    discovery_dates=meta["discovery_dates"], check_dates=meta["check_dates"], adaptive_study=adaptive_root.name,
                    adaptive_metadata_sha256=file_hash(adaptive_root / "study.json"), source_manifest=meta["source_manifest"],
                    protocol_sha256=file_hash(protocol),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 反转采样相位敏感性", "", "所有相位如实报告，不挑最佳相位。每个组合独立模拟，禁止跨方案直接合计损益。",
        "", "```", summary.to_string(index=False), "```", "", "此前数据已探索，结果不能作为独立验证。", ""]), encoding="utf-8")
    print(summary.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--adaptive-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("reversal_phase_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, args.adaptive_root, output)


if __name__ == "__main__":
    main()
