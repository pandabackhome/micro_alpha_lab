"""First crossing of a past-date stock threshold, independent of minute phase."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.adaptive_exit_search import underlying_exit_scheduler
from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.stock_strategy_search import COLUMNS, aggregate, calibrate, fresh_gate, pick, stock_observations


def crossing_signals(calibrated, mode="first_crossing"):
    if mode not in ("first_crossing", "recovery", "recovery_turn"):
        raise ValueError("unknown event mode")
    result = calibrated.sort_values("timestamp").copy()
    if not result.timestamp.diff().dropna().eq(pd.Timedelta(seconds=1)).all():
        raise ValueError("event detection requires a complete one-second observation grid")
    valid = result.ready & result.valid300 & result.seconds_from_open.lt(7200)
    outside = result.r60.abs().gt(result.threshold_r60)
    flipped = np.sign(result.r60).ne(np.sign(result.r60.shift(1)))
    eligible = valid & valid.shift(1).fillna(False)
    if mode == "first_crossing":
        first = eligible & outside & (~outside.shift(1).fillna(False) | flipped)
    else:
        first = eligible & ~outside & outside.shift(1).fillna(False) & ~flipped
        if mode == "recovery_turn":
            first &= (result.r60 * result.r5).lt(0)
    result["scheduled"] = first
    result["direction"] = (-np.sign(result.r60)).where(first, 0).fillna(0).astype(int)
    result["signal_status"] = np.where(first, "signal", "no_signal")
    result["regime"] = mode
    result["factor"] = mode
    result["previous_r60"] = result.r60.shift(1)
    result["previous_threshold_r60"] = result.threshold_r60.shift(1)
    return result


def run(data_root, output, mode="first_crossing"):
    settings = replace(StudySettings(), holding_seconds=300)
    paths = sorted((data_root / "features").glob("*.parquet"))
    if len(paths) < 14:
        raise ValueError("requires at least fourteen dates")
    dates = [p.stem for p in paths]
    discovery_dates, check_dates = dates[5:10], dates[10:]
    output.mkdir(parents=True, exist_ok=False)
    history, daily, ledgers, manifests, all_signals = [], [], [], [], []
    for path in paths:
        frame = pd.read_parquet(path, columns=COLUMNS)
        ordinary = stock_observations(frame, settings)
        if set(ordinary.date) != {path.stem}:
            raise ValueError("feature date mismatch")
        manifests.append(dict(file="features/" + path.name, sha256=file_hash(path)))
        if len(history) >= 5:
            dense = stock_observations(frame, settings, sampling_seconds=1)
            signals = crossing_signals(calibrate(dense, history, settings), mode=mode)
            all_signals.append(signals[signals.scheduled])
            source = data_root / "normalized" / path.stem / "events.parquet"
            books = read_books(source, path.stem)
            manifests.append(dict(file="normalized/" + path.stem + "/events.parquet", sha256=file_hash(source)))
            for exit_style in ("baseline", "stop_half"):
                scheduler = None if exit_style == "baseline" else underlying_exit_scheduler(frame, None, .5)
                for latency in (1, 5):
                    factor = exit_style + "_lag" + str(latency)
                    execution = replace(settings, latency_seconds=latency)
                    ledger = simulate_single_day(signals.assign(factor=factor), books, execution,
                                                 entry_gate=fresh_gate(books), exit_scheduler=scheduler)
                    ledger["horizon"] = 300
                    ledgers.append(ledger)
                    metrics = execution_metrics(ledger)
                    metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
                    daily.append(dict(date=path.stem, factor=factor, horizon=300, **metrics))
            print(json.dumps(dict(date=path.stem, crossing_events=int(signals.scheduled.sum()))), flush=True)
        history.append(ordinary)
    daily = pd.DataFrame(daily)
    discovery = aggregate(daily, discovery_dates)
    selected = pick(discovery)
    checking = aggregate(daily, check_dates).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", checking),
                          ("all_dates", aggregate(daily, dates[5:]))]:
        values.to_csv(output / (name + ".csv"), index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(output / "opportunities.parquet", index=False)
    pd.concat(all_signals, ignore_index=True).to_parquet(output / "signals.parquet", index=False)
    package = Path(__file__).resolve().parents[2]
    protocol = package / ("research/event_reversal_protocol.md" if mode == "first_crossing" else "research/event_recovery_protocol.md")
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    metadata = dict(study="stock_reversal_event", mode=mode, settings=asdict(settings), trials=4,
                    dates=dates, discovery_dates=discovery_dates, check_dates=check_dates,
                    source_manifest=manifests, protocol_sha256=file_hash(protocol),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 正股逐秒反转事件：" + mode, "", "固定阈值、止损倍数与期限，不根据结果挑采样秒数。所有日期均已探索。",
        "", "## 前段全部结果", "", "```", discovery.to_string(index=False), "```", "",
        "## 前段选出的候选在后段的结果", "", "```", checking.to_string(index=False), "```", ""]), encoding="utf-8")
    print(discovery.to_string(index=False))
    print("Selected check:"); print(checking.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("first_crossing", "recovery", "recovery_turn"), default="first_crossing")
    args = parser.parse_args(argv)
    output = Path("results/research") / ("event_reversal_" + args.mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, output, mode=args.mode)


if __name__ == "__main__":
    main()
