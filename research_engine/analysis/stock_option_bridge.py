"""Evaluation-only bridge from stock signals to executed option outcomes."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import FEATURE_COLUMNS, StudySettings, file_hash
from research_engine.analysis.directional_factors import daily_ci, future_outcomes


def run(archive_data, local_data, phase_root, new_root, output):
    phase = pd.read_parquet(phase_root / "opportunities.parquet")
    phase = phase[phase.exit_style.eq("baseline") & phase.latency.eq(1)].copy()
    phase["sample"] = "old_exploration"
    new = pd.read_parquet(new_root / "opportunities.parquet")
    new = new[new.factor.eq("clock_reversal")].copy()
    new["phase"], new["sample"] = 30, "new_frozen"
    signals = pd.concat([phase, new], ignore_index=True)
    signals = signals[signals.signal_status.eq("signal")].copy()
    output.mkdir(parents=True, exist_ok=False)
    labels, daily, trades, sources = [], [], [], []
    for day, group in signals.groupby("date"):
        origin = "local" if group["sample"].eq("new_frozen").any() else "archive"
        root = local_data if origin == "local" else archive_data
        path = root / "features" / (day + ".parquet")
        frame = pd.read_parquet(path, columns=FEATURE_COLUMNS)
        sources.append(dict(origin=origin, file="features/" + path.name, sha256=file_hash(path)))
        for horizon in (60, 180, 300):
            future = future_outcomes(frame, replace(StudySettings(), holding_seconds=horizon))
            joined = group.merge(future, on="timestamp", validate="many_to_one")
            joined["label_horizon"] = horizon
            joined["signed_bps"] = joined.direction * joined.delayed_return_bps
            labels.append(joined[["date", "timestamp", "sample", "phase", "direction", "status", "label_horizon", "signed_bps"]])
            for (sample, phase_value), panel in joined.groupby(["sample", "phase"]):
                for universe in ("all_intents", "filled_positions"):
                    selected = panel if universe == "all_intents" else panel[panel.status.eq("closed")]
                    valid = selected[selected.signed_bps.notna()]
                    daily.append(dict(date=day, sample=sample, phase=phase_value, horizon=horizon, universe=universe,
                                      signals=len(selected), labelled=len(valid), mean_signed_bps=valid.signed_bps.mean()))
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns")
        frame = frame.set_index("timestamp")
        good = frame.qqq_depth_age_seconds.between(0,5) & frame.qqq_bid.gt(0) & frame.qqq_ask.ge(frame.qqq_bid)
        mid = ((frame.qqq_bid + frame.qqq_ask)/2).where(good)
        for trade in group[group.status.eq("closed")].to_dict("records"):
            start, end = mid.get(trade["entry_time"], np.nan), mid.get(trade["exit_time"], np.nan)
            trade["actual_stock_signed_bps"] = trade["direction"] * (end/start-1) * 10000
            trade["actual_stock_signed_dollars"] = trade["direction"] * (end-start)
            trades.append(trade)
    daily = pd.DataFrame(daily)
    summaries = []
    for (sample, phase_value, horizon, universe), group in daily.groupby(["sample", "phase", "horizon", "universe"]):
        lo, hi = daily_ci(group.mean_signed_bps)
        summaries.append(dict(sample=sample, phase=phase_value, horizon=horizon, universe=universe,
                              signals=int(group.signals.sum()), labelled=int(group.labelled.sum()), days=len(group),
                              positive_days=int(group.mean_signed_bps.gt(0).sum()),
                              daily_mean_bps=group.mean_signed_bps.mean(), ci_low=lo, ci_high=hi))
    trades = pd.DataFrame(trades)
    bridges = []
    for (sample, phase_value), group in trades.groupby(["sample", "phase"]):
        bridges.append(dict(sample=sample, phase=phase_value, closed=len(group),
                            mean_stock_signed_bps=group.actual_stock_signed_bps.mean(),
                            mean_option_mid_pnl=group.mid_pnl.mean(), mean_spread_cost=group.spread_cost.mean(),
                            mean_slippage=group.slippage_cost.mean(), mean_commission=group.commission.mean(),
                            mean_net_pnl=group.net_pnl.mean(),
                            stock_dollar_option_mid_correlation=group.actual_stock_signed_dollars.corr(group.mid_pnl)))
    for name, values in [("stock_daily", daily), ("stock_summary", pd.DataFrame(summaries)), ("execution_bridge", pd.DataFrame(bridges))]:
        values.to_csv(output / (name + ".csv"), index=False)
    pd.concat(labels, ignore_index=True).to_parquet(output / "evaluation_labels.parquet", index=False)
    trades.to_parquet(output / "executed_trade_evaluation.parquet", index=False)
    metadata = dict(study="evaluation_only_stock_option_bridge", source_manifest=sources,
                    phase_study=phase_root.name, phase_metadata_sha256=file_hash(phase_root / "study.json"),
                    new_study=new_root.name, new_metadata_sha256=file_hash(new_root / "study.json"),
                    code_sha256=file_hash(Path(__file__)), future_labels_used_for_decisions=False)
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 从正股信号到期权成交的诊断", "", "这是结果解释，不生成新交易规则。旧日期四个采样相位全部显示；新增日期只解释已冻结的第30秒方案。",
        "所有意向信号与实际成功开仓的信号分开计算。意向信号可能重叠，不能把其均值当作可实现组合收益。",
        "按日期计算区间，未校正前面多轮探索；两个新增日期的区间仅为描述。", "",
        "## 正股方向", "", "```", pd.DataFrame(summaries).to_string(index=False), "```", "",
        "## 已成交交易的价格变化与成本", "", "```", pd.DataFrame(bridges).to_string(index=False), "```", "",
        "正股变化与期权中价损益的相关性只说明这些成交的共变关系，不代表预测能力。没有把其余价格变化归因为单一Theta或隐含波动因素。", ""]), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(bridges).to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data", type=Path, required=True)
    parser.add_argument("--local-data", type=Path, default=Path("data"))
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("stock_option_bridge_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data, args.local_data, args.phase_root, args.new_root, output)


if __name__ == "__main__":
    main()
