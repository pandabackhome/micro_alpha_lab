"""Replay three frozen rules on newly discovered, previously unused dates."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.adaptive_exit_search import underlying_exit_scheduler
from research_engine.analysis.causal_volatility import BOOK_COLUMNS, StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, future_outcomes, simulate_single_day
from research_engine.analysis.event_reversal_search import crossing_signals
from research_engine.analysis.reversal_execution_audit import audit_raw_fills
from research_engine.analysis.search_artifact_audit import validate_ledger
from research_engine.analysis.stock_strategy_search import COLUMNS, aggregate, calibrate, fresh_gate, signals_for, stock_observations


def run(archive_data, recordings_root, local_data, freeze_root, output):
    freeze = json.loads((freeze_root / "freeze.json").read_text())
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/new_dates_protocol.md"
    if file_hash(protocol) != freeze["protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    for record in freeze["recordings"]:
        if file_hash(recordings_root / record["file"]) != record["compressed_sha256"]:
            raise ValueError("frozen recording changed")
    settings = replace(StudySettings(), holding_seconds=300)
    output.mkdir(parents=True, exist_ok=False)
    history_records = json.loads((freeze_root / "history_manifest.json").read_text())
    old_paths = []
    for record in history_records:
        path = archive_data / record["file"]
        if file_hash(path) != record["sha256"]:
            raise ValueError("frozen calibration file changed")
        old_paths.append(path)
    old_paths = sorted(old_paths)
    if len(old_paths) != 5:
        raise ValueError("five calibration dates required")
    history, manifests = [], []
    for path in old_paths:
        history.append(stock_observations(pd.read_parquet(path, columns=COLUMNS), settings))
        manifests.append(dict(origin="archive", file="features/" + path.name, sha256=file_hash(path)))
    daily, ledgers, signals_saved, quality, direction, audits = [], [], [], [], [], []
    for day in sorted(freeze["dates"]):
        feature_path = local_data / "features" / (day + ".parquet")
        normalized_path = local_data / "normalized" / day / "events.parquet"
        frame = pd.read_parquet(feature_path, columns=COLUMNS)
        observed = stock_observations(frame, settings)
        if set(observed.date) != {day}:
            raise ValueError("local feature date mismatch")
        normalized_meta = json.loads(normalized_path.with_suffix(".parquet.meta.json").read_text())
        contracts = normalized_meta["recording_metadata"]["contracts"]
        if contracts["underlying"] != "QQQ.US" or contracts["expiry"] != day:
            raise ValueError("recording underlying or expiry mismatch")
        book = read_books(normalized_path, day)
        raw = pd.read_parquet(normalized_path, columns=BOOK_COLUMNS,
                              filters=[("kind", "=", "depth"), ("symbol", "in", list(book.books))])
        spot_valid = (frame.qqq_bid.gt(0) & frame.qqq_ask.ge(frame.qqq_bid) & frame.qqq_depth_age_seconds.between(0, 5) &
                      np.isfinite(frame.qqq_bid) & np.isfinite(frame.qqq_ask))
        quality.append(dict(date=day, feature_rows=len(frame), fresh_stock_fraction=float(spot_valid.mean()),
                            valid_300_fraction=float(observed.valid300.mean()), known_option_contracts=len(book.contracts),
                            option_depth_rows=len(raw), first_option_quote=raw.available_at.min(),
                            last_option_quote=raw.available_at.max(), history_dates="|".join(p.date.iloc[0] for p in history[-5:])))
        grid = signals_for(calibrate(observed, history, settings), "morning")
        grid = grid[grid.factor.eq("reversal60")].copy()
        event = crossing_signals(calibrate(stock_observations(frame, settings, sampling_seconds=1), history, settings), mode="recovery")
        configurations = [("clock_reversal", grid, None),
                          ("clock_reversal_stop", grid, underlying_exit_scheduler(frame, None, .5)),
                          ("recovery_reversal", event, None)]
        for name, signals, scheduler in configurations:
            signals = signals.assign(factor=name)
            signals_saved.append(signals[signals.scheduled])
            ledger = simulate_single_day(signals, book, settings, entry_gate=fresh_gate(book), exit_scheduler=scheduler)
            ledger["horizon"] = 300
            validate_ledger(ledger, settings.multiplier)
            raw_checked = audit_raw_fills(ledger, raw, settings)
            audits.append(dict(date=day, factor=name, raw_quotes_checked=raw_checked))
            metrics = execution_metrics(ledger)
            metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
            daily.append(dict(date=day, factor=name, horizon=300, **metrics))
            ledgers.append(ledger)
            intended = signals[signals.scheduled & signals.signal_status.eq("signal")]
            for horizon in (60, 180, 300):
                labels = future_outcomes(frame, replace(settings, holding_seconds=horizon))
                joined = intended.merge(labels, on="timestamp", validate="one_to_one")
                valid = joined[joined.delayed_return_bps.notna()]
                signed = valid.direction * valid.delayed_return_bps
                direction.append(dict(date=day, factor=name, horizon=horizon, intended=len(intended), labelled=len(valid),
                                      mean_signed_bps=signed.mean(), hit_rate=signed.gt(0).mean()))
        history.append(observed)
        manifests += [dict(origin="local", file="features/" + day + ".parquet", sha256=file_hash(feature_path)),
                      dict(origin="local", file="normalized/" + day + "/events.parquet", sha256=file_hash(normalized_path))]
        print("Validated frozen rules on " + day, flush=True)
    daily = pd.DataFrame(daily)
    summary = aggregate(daily, freeze["dates"])
    for name, values in [("daily", daily), ("summary", summary), ("data_quality", pd.DataFrame(quality)),
                          ("stock_direction", pd.DataFrame(direction)), ("raw_quote_audit", pd.DataFrame(audits))]:
        values.to_csv(output / (name + ".csv"), index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(output / "opportunities.parquet", index=False)
    pd.concat(signals_saved, ignore_index=True).to_parquet(output / "signals.parquet", index=False)
    (output / "protocol.md").write_bytes(protocol.read_bytes())
    (output / "freeze.json").write_bytes((freeze_root / "freeze.json").read_bytes())
    metadata = dict(study="frozen_rules_new_dates", dates=freeze["dates"], settings=asdict(settings),
                    protocol_sha256=file_hash(protocol), source_manifest=manifests,
                    freeze_sha256=file_hash(freeze_root / "freeze.json"),
                    history_manifest_sha256=file_hash(freeze_root / "history_manifest.json"),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 新增9月16、17日：冻结规则检验", "", "这两日未参与之前的规则搜索。生成特征前已冻结三条规则，未按新增结果改参数。",
        "仍只是两个新增历史日期，不等同于真实盘中前瞻测试。每张美元损益含价差、手续费及滑点。", "",
        "## 两日合计", "", "```", summary.to_string(index=False), "```", "",
        "## 逐日", "", "```", daily.to_string(index=False), "```", "",
        "## 数据覆盖与历史校准日期", "", "```", pd.DataFrame(quality).to_string(index=False), "```", "",
        "## 正股方向，仅评估用", "", "```", pd.DataFrame(direction).to_string(index=False), "```", "",
        "每笔入场和退出都独立核对原始归一化报价，核对次数见 raw_quote_audit.csv。事件信号可能相互重叠，信号数不是独立样本数。", ""]), encoding="utf-8")
    print(summary.to_string(index=False))
    print(daily[["date", "factor", "closed", "realized_net_pnl", "unclosed"]].to_string(index=False))
    print(pd.DataFrame(quality).to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data", type=Path, required=True)
    parser.add_argument("--recordings-root", type=Path, required=True)
    parser.add_argument("--local-data", type=Path, default=Path("data"))
    parser.add_argument("--freeze-root", type=Path, default=Path("results/research/new_dates_frozen"))
    args = parser.parse_args(argv)
    output = Path("results/research") / ("new_dates_validation_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data, args.recordings_root, args.local_data, args.freeze_root, output)


if __name__ == "__main__":
    main()
