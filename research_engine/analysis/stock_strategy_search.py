"""Underlying-only strategy library with minute-scale, costed option execution."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import (
    FEATURE_COLUMNS, StudySettings, observations, read_books, file_hash,
)
from research_engine.analysis.directional_factors import daily_ci, execution_metrics, mean_or_none, simulate_single_day
from research_engine.config import load_config, resolve_path

RULES = {
    "momentum60": "一分钟强趋势", "momentum180": "三分钟强趋势",
    "reversal60": "一分钟强变动反转", "reversal180": "三分钟强变动反转",
    "volume_breakout": "放量突破", "failed_breakout": "突破失败回归",
    "trend_pullback": "趋势回撤再启动", "compression_breakout": "窄幅整理后突破",
    "band_reversal": "偏离均价后回归", "flow_divergence": "成交压力与价格背离",
    "flow_trend": "成交压力确认趋势", "opening_breakout": "开盘区间突破",
}
HORIZONS = (60, 180, 300)
CONTEXTS = ("all", "high_vol", "morning", "low_spread")
COLUMNS = FEATURE_COLUMNS + ["bid_size", "ask_size", "buy_volume_30s", "sell_volume_30s", "total_volume_5s"]
CALIBRATION = {"r60": (.75, True), "r180": (.75, True), "volume5": (.75, False),
               "range60": (.25, False), "deviation": (.75, True), "flow30": (.75, True),
               "past_vol": (.75, False), "spread": (.5, False)}


def stock_observations(frame, settings, sampling_seconds=None):
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    base = observations(frame, settings, sampling_seconds=sampling_seconds)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns")
    good = (frame.qqq_depth_age_seconds.between(0, settings.max_quote_age_seconds) &
            frame.qqq_bid.gt(0) & frame.qqq_ask.ge(frame.qqq_bid) &
            np.isfinite(frame.qqq_bid) & np.isfinite(frame.qqq_ask))
    mid = ((frame.qqq_bid + frame.qqq_ask) / 2).where(good)
    depth = (frame.bid_size + frame.ask_size).where(frame.bid_size.gt(0) & frame.ask_size.gt(0))
    values = pd.DataFrame({"timestamp": frame.timestamp, "valid300": good.rolling(301, min_periods=301).sum().eq(301),
                           "spread": frame.qqq_ask - frame.qqq_bid, "volume5": frame.total_volume_5s,
                           "high60": mid.shift(1).rolling(60, min_periods=60).max(),
                           "low60": mid.shift(1).rolling(60, min_periods=60).min(),
                           "old_high": mid.shift(31).rolling(120, min_periods=120).max(),
                           "old_low": mid.shift(31).rolling(120, min_periods=120).min(),
                           "recent_high": mid.rolling(30, min_periods=30).max(),
                           "recent_low": mid.rolling(30, min_periods=30).min(),
                           "deviation": mid / mid.rolling(180, min_periods=180).mean() - 1,
                           "flow30": (frame.buy_volume_30s - frame.sell_volume_30s) / depth.rolling(30, min_periods=30).mean()})
    for window in (5, 30, 60, 180):
        values["r" + str(window)] = mid / mid.shift(window) - 1
    values["range60"] = (values.high60 - values.low60) / mid
    local = frame.timestamp.dt.tz_convert("America/New_York")
    seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second - 34200
    values["seconds_from_open"] = seconds
    opening = mid[(seconds >= 0) & (seconds < 900)]
    covered = opening.notna().sum() >= 855
    values["opening_high"] = np.where((seconds >= 900) & covered, opening.max(), np.nan)
    values["opening_low"] = np.where((seconds >= 900) & covered, opening.min(), np.nan)
    return base.merge(values, on="timestamp", validate="one_to_one")


def calibrate(current, history, settings):
    result = current.copy()
    result["history_last_date"] = None
    result["ready"] = False
    for field in CALIBRATION:
        result["threshold_" + field] = np.nan
    if history:
        past = pd.concat(history[-settings.calibration_days:], ignore_index=True)
        if past.date.max() >= current.date.min():
            raise ValueError("history must contain only earlier dates")
        if past.date.nunique() != min(len(history), settings.calibration_days):
            raise ValueError("distinct historical dates required")
        result["history_last_date"] = past.date.max()
    if len(history) < settings.calibration_days:
        return result
    for bucket, idx in result.groupby("half_hour").groups.items():
        group = past[past.half_hour.eq(bucket) & past.valid300]
        for field, (quantile, absolute) in CALIBRATION.items():
            values = group[field].replace([np.inf, -np.inf], np.nan).dropna()
            if absolute:
                values = values.abs()
            if len(values) >= settings.min_history_samples:
                result.loc[idx, "threshold_" + field] = values.quantile(quantile)
    result["ready"] = result[["threshold_" + field for field in CALIBRATION]].notna().all(axis=1)
    return result


def signals_for(current, context):
    if context not in CONTEXTS:
        raise ValueError("unknown context")
    p = current.spot
    r5, r30, r60, r180 = (current["r" + str(w)] for w in (5, 30, 60, 180))
    strong60, strong180 = r60.abs().gt(current.threshold_r60), r180.abs().gt(current.threshold_r180)
    volume = current.volume5.gt(current.threshold_volume5)
    up, down = p.gt(current.high60), p.lt(current.low60)
    flow = current.flow30
    strong_flow = flow.abs().gt(current.threshold_flow30)
    directions = {
        "momentum60": np.sign(r60).where(strong60, 0),
        "momentum180": np.sign(r180).where(strong180, 0),
        "reversal60": -np.sign(r60).where(strong60, 0),
        "reversal180": -np.sign(r180).where(strong180, 0),
        "volume_breakout": pd.Series(np.select([volume & up, volume & down], [1, -1], 0), index=current.index),
        "failed_breakout": pd.Series(np.select([
            current.recent_high.gt(current.old_high + current.spread) & p.lt(current.old_high) & r5.lt(0),
            current.recent_low.lt(current.old_low - current.spread) & p.gt(current.old_low) & r5.gt(0)], [-1, 1], 0), index=current.index),
        "trend_pullback": np.sign(r180).where(strong180 & (r180 * r30).lt(0) & (r180 * r5).gt(0), 0),
        "compression_breakout": pd.Series(np.select([volume & up, volume & down], [1, -1], 0), index=current.index).where(current.range60.lt(current.threshold_range60), 0),
        "band_reversal": -np.sign(current.deviation).where(current.deviation.abs().gt(current.threshold_deviation) & (current.deviation * r5).lt(0), 0),
        "flow_divergence": np.sign(r30).where(strong_flow & (flow * r30).lt(0), 0),
        "flow_trend": np.sign(r60).where(strong60 & strong_flow & (flow * r60).gt(0), 0),
        "opening_breakout": pd.Series(np.select([volume & p.gt(current.opening_high), volume & p.lt(current.opening_low)], [1, -1], 0), index=current.index).where(current.seconds_from_open.between(900, 3599), 0),
    }
    allowed = pd.Series(True, index=current.index)
    if context == "high_vol":
        allowed = current.past_vol.gt(current.threshold_past_vol)
    elif context == "morning":
        allowed = current.seconds_from_open.lt(7200)
    elif context == "low_spread":
        allowed = current.spread.le(current.threshold_spread)
    frames = []
    for rule, direction in directions.items():
        result = current.copy()
        result["factor"] = rule
        result["context"] = context
        result["direction"] = direction.fillna(0).astype(int)
        result["signal_status"] = np.where(result.direction.ne(0), "signal", "no_signal")
        result.loc[result.direction.ne(0) & ~allowed, "signal_status"] = "context_filtered"
        result.loc[~result.ready, "signal_status"] = "insufficient_history"
        result.loc[~result.valid300, "signal_status"] = "invalid_spot_window"
        result.loc[~result.signal_status.eq("signal"), "direction"] = 0
        result["regime"] = "research"
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def fresh_gate(books):
    def gate(signal, symbol, when):
        quote, reason = books.quote(symbol, when, 1.)
        if quote is None:
            return False, "fresh_entry:" + reason, {}
        if when - pd.Timedelta(seconds=quote["age"]) <= signal["timestamp"]:
            return False, "quote_not_updated_after_signal", {}
        return True, "ok", {"checked_entry_age": quote["age"]}
    return gate


def aggregate(daily, dates):
    rows = []
    for (rule, horizon), group in daily[daily.date.isin(dates)].groupby(["factor", "horizon"], sort=False):
        closed = int(group.closed.sum())
        realized = float(group.realized_net_pnl.sum())
        stress = float(group.net_with_unclosed_zero_recovery.sum())
        lo, hi = daily_ci(group.mean_net_pnl)
        rows.append(dict(factor=rule, horizon=horizon, signals=int(group.policy_signals.sum()), closed=closed,
                         mean_net_pnl=realized / closed if closed else None, realized_net_pnl=realized,
                         unclosed=int(group.unclosed.sum()), unclosed_debit=float(group.unclosed_debit.sum()),
                         stress_net=stress, positive_days=int(group.net_with_unclosed_zero_recovery.gt(0).sum()),
                         active_days=int(group.closed.gt(0).sum()), days=len(dates),
                         without_best_day=stress - float(group.net_with_unclosed_zero_recovery.max()),
                         daily_mean=mean_or_none(group.mean_net_pnl), daily_ci_low=lo, daily_ci_high=hi,
                         entry_rejected=int(group.entry_rejected.sum())))
    return pd.DataFrame(rows)


def pick(discovery):
    eligible = discovery[(discovery.closed >= 30) & (discovery.active_days >= 4) & (discovery.positive_days >= 3) &
                         discovery.stress_net.gt(0) & discovery.without_best_day.gt(0)]
    return eligible.sort_values(["without_best_day", "factor", "horizon"], ascending=[False, True, True]).head(3)


def markdown(frame):
    columns = ["factor", "horizon", "closed", "mean_net_pnl", "stress_net", "positive_days", "without_best_day"]
    lines = ["| 规则 | 持有秒 | 已退出 | 每笔净损益 | 压力总损益 | 正收益天数 | 省略最佳日后 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in frame.to_dict("records"):
        values = []
        for col in columns:
            value = row[col]
            values.append(RULES[value] if col == "factor" else ("—" if pd.isna(value) else ("{:.4f}".format(value) if isinstance(value, float) else str(value))))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def run(data_root, output, context="all", strict_entry=False):
    settings = StudySettings()
    paths = sorted((data_root / "features").glob("*.parquet"))
    if len(paths) < 14:
        raise ValueError("this discovery/check protocol requires at least 14 dates")
    evaluation_dates = [path.stem for path in paths[5:]]
    discovery_dates, check_dates = evaluation_dates[:5], evaluation_dates[5:]
    output.mkdir(parents=True, exist_ok=False)
    (output / "opportunities").mkdir()
    protocol = Path(__file__).resolve().parents[2] / "research/stock_strategy_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    history, daily, manifests, counts = [], [], [], []
    for path in paths:
        frame = pd.read_parquet(path, columns=COLUMNS)
        current = stock_observations(frame, settings)
        if set(current.date) != {path.stem}:
            raise ValueError("feature date mismatch")
        calibrated = calibrate(current, history, settings)
        signals = signals_for(calibrated, context)
        history.append(current)
        manifests.append(dict(file="features/" + path.name, sha256=file_hash(path)))
        if path.stem not in evaluation_dates:
            continue
        source = data_root / "normalized" / path.stem / "events.parquet"
        books = read_books(source, path.stem)
        manifests.append(dict(file="normalized/" + path.stem + "/events.parquet", sha256=file_hash(source)))
        day_ledgers = []
        for rule in RULES:
            selected = signals[signals.factor.eq(rule)]
            for horizon in HORIZONS:
                execution = replace(settings, holding_seconds=horizon)
                ledger = simulate_single_day(selected, books, execution, entry_gate=fresh_gate(books) if strict_entry else None)
                ledger["horizon"] = horizon
                day_ledgers.append(ledger)
                metrics = execution_metrics(ledger)
                metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
                daily.append(dict(date=path.stem, factor=rule, horizon=horizon, **metrics))
        day_ledger = pd.concat(day_ledgers, ignore_index=True)
        day_ledger.to_parquet(output / "opportunities" / (path.stem + ".parquet"), index=False)
        counts.append(day_ledger.groupby(["date", "factor", "horizon", "status"]).size().rename("count").reset_index())
        print(json.dumps({"date": path.stem, "candidates": len(day_ledger), "closed": int(day_ledger.status.eq("closed").sum())}), flush=True)
    daily = pd.DataFrame(daily)
    discovery = aggregate(daily, discovery_dates)
    selected = pick(discovery)
    checking = aggregate(daily, check_dates)
    finalists = checking.merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    overall = aggregate(daily, evaluation_dates)
    for name, data in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", finalists), ("all_dates", overall)]:
        data.to_csv(output / (name + ".csv"), index=False)
    pd.concat(counts).to_csv(output / "status_counts.csv", index=False)
    pd.concat(history, ignore_index=True).to_parquet(output / "observations.parquet", index=False)
    metadata = dict(study="underlying_strategy_search_v1", rules=RULES, horizons=HORIZONS, context=context,
                    strict_entry=strict_entry, settings=asdict(settings), dates=[path.stem for path in paths],
                    discovery_dates=discovery_dates, check_dates=check_dates, trials=len(RULES) * len(HORIZONS),
                    protocol_sha256=file_hash(protocol), source_manifest=manifests,
                    code_manifest=[dict(file=str(path.relative_to(protocol.parent.parent)), sha256=file_hash(path))
                                   for path in sorted((protocol.parent.parent / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 正股策略开仓期权搜索", "", "条件：{}；严格新鲜入场：{}。本轮36组合，历史探索，非独立新数据。".format(context, strict_entry),
             "每笔一张，净损益含价差、手续费和滑点。压力总损益另扣未平仓入场资金，非实际结算。", "",
             "## 前五个评估日的全部结果", ""] + markdown(discovery)
    lines += ["", "## 按预定条件筛出的候选", ""] + markdown(selected)
    lines += ["", "## 候选的后段检验", ""] + markdown(finalists)
    lines += ["", "无候选时保留空表，不自动放宽条件。完整逐日记录与所有失败状态保存在 daily.csv、status_counts.csv 和 opportunities/。",
              "所有日期此前已经探索；多策略、多期限筛选会提高偶然盈利的概率。按日区间和省略最佳日只用于描述，不能代替新增样本。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Discovery top rows:")
    print(discovery.sort_values("stress_net", ascending=False).head(8).to_string(index=False))
    print("Selected candidates:")
    print(finalists.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--context", choices=CONTEXTS, default="all")
    parser.add_argument("--strict-entry", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_root or resolve_path(load_config(), "results")
    output = root / ("stock_strategy_" + args.context + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root.expanduser(), output, args.context, args.strict_entry)


if __name__ == "__main__":
    main()
