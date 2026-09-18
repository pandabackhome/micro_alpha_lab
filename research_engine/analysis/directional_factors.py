"""Past-date factor thresholds, single-leg option execution, and volatility filters."""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import (
    FEATURE_COLUMNS, GROUPS, ContractBooks, StudySettings, assign_regimes,
    file_hash, observations, read_books,
)
from research_engine.config import load_config, resolve_path


FACTORS = {
    "ofi_5s": "5 秒 OFI",
    "signed_volume_5s": "5 秒主动买卖量差",
    "depth_imbalance": "盘口深度不平衡",
}
INPUT_COLUMNS = FEATURE_COLUMNS + ["ofi_5s", "buy_volume_5s", "sell_volume_5s", "depth_imbalance"]
POLICIES = ("all", "high_only")


def factor_observations(frame, settings):
    """Only observable features enter the history used to set thresholds."""
    base = observations(frame, settings)
    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True).dt.as_unit("ns")
    source["signed_volume_5s"] = source["buy_volume_5s"] - source["sell_volume_5s"]
    return base.merge(source[["timestamp"] + list(FACTORS)], on="timestamp", how="left", validate="one_to_one")


def factor_signals(current, history, settings):
    """Positive upper tail buys CALL; negative lower tail buys PUT; signs never fitted."""
    base = assign_regimes(current, history, settings)
    selected = history[-settings.calibration_days:]
    past = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    panels = []
    for factor in FACTORS:
        result = base.drop(columns=list(FACTORS)).copy()
        result["factor"] = factor
        result["factor_value"] = current[factor].to_numpy()
        result["factor_q25"] = np.nan
        result["factor_q75"] = np.nan
        result["factor_history_samples"] = 0
        result["direction"] = 0
        result["signal_status"] = result["regime"].where(~result["regime"].isin(GROUPS), "insufficient_factor_history")
        if len(selected) == settings.calibration_days:
            for bucket, indices in result.groupby("half_hour").groups.items():
                values = past.loc[past["half_hour"].eq(bucket) & past["past_vol"].notna(), factor]
                values = values.replace([np.inf, -np.inf], np.nan).dropna()
                result.loc[indices, "factor_history_samples"] = len(values)
                if len(values) < settings.min_history_samples:
                    continue
                q25, q75 = values.quantile([0.25, 0.75]).to_numpy()
                result.loc[indices, ["factor_q25", "factor_q75"]] = (q25, q75)
                eligible = result.index.isin(indices) & result["regime"].isin(GROUPS)
                result.loc[eligible, "signal_status"] = "no_signal"
                bullish = eligible & result["factor_value"].gt(max(0.0, q75))
                bearish = eligible & result["factor_value"].lt(min(0.0, q25))
                result.loc[bullish, "direction"] = 1
                result.loc[bearish, "direction"] = -1
                result.loc[bullish | bearish, "signal_status"] = "signal"
        invalid = result["regime"].isin(GROUPS) & ~np.isfinite(result["factor_value"])
        result.loc[invalid, "signal_status"] = "invalid_factor"
        result.loc[invalid, "direction"] = 0
        panels.append(result)
    return pd.concat(panels, ignore_index=True)


def future_outcomes(frame, settings):
    """Evaluation-only labels, kept outside threshold history and execution inputs."""
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    stamps = pd.to_datetime(frame["timestamp"], utc=True).dt.as_unit("ns")
    bid, ask = frame["qqq_bid"], frame["qqq_ask"]
    good = (frame["qqq_depth_age_seconds"].between(0, settings.max_quote_age_seconds) &
            bid.gt(0) & ask.ge(bid) & np.isfinite(bid) & np.isfinite(ask))
    mid = ((bid + ask) / 2).where(good)
    lag, hold = settings.latency_seconds, settings.holding_seconds
    return pd.DataFrame({
        "timestamp": stamps,
        "future_return_bps": (mid.shift(-hold) / mid - 1) * 10000,
        "delayed_return_bps": (mid.shift(-lag - hold) / mid.shift(-lag) - 1) * 10000,
    })


def mean_or_none(values):
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if len(clean) else None


def daily_ci(values):
    values = np.asarray(pd.Series(values, dtype=float).dropna(), float)
    if len(values) < 2:
        return None, None
    rng = np.random.RandomState(42)
    sample = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(sample, [0.025, 0.975]))


def rank_ic(frame):
    pair = frame[["factor_value", "future_return_bps"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or (pair.nunique() < 2).any():
        return None
    return float(pair["factor_value"].corr(pair["future_return_bps"], method="spearman"))


def direction_metrics(group):
    selected = group[group["signal_status"].eq("signal")]
    labelled = selected[selected["future_return_bps"].notna()]
    delayed = selected[selected["delayed_return_bps"].notna()]
    signed = labelled["direction"] * labelled["future_return_bps"]
    return {
        "observations": len(group), "signals": len(selected), "labelled_signals": len(labelled),
        "missing_future_labels": len(selected) - len(labelled),
        "call_signals": int(selected["direction"].eq(1).sum()),
        "put_signals": int(selected["direction"].eq(-1).sum()),
        "spearman_ic": rank_ic(group), "hit_rate": mean_or_none(signed > 0),
        "flat_rate": mean_or_none(labelled["future_return_bps"].eq(0)),
        "random_sign_hit_baseline": mean_or_none(labelled["future_return_bps"].ne(0) * 0.5),
        "mean_signed_return_bps": mean_or_none(signed),
        "delayed_labelled_signals": len(delayed),
        "delayed_hit_rate": mean_or_none(delayed["direction"] * delayed["delayed_return_bps"] > 0),
        "mean_delayed_signed_return_bps": mean_or_none(delayed["direction"] * delayed["delayed_return_bps"]),
    }


def summarize_direction(panel):
    summaries, daily = [], []
    eligible = panel[panel["signal_status"].isin(["signal", "no_signal"]) & panel["scheduled"]]
    for factor in FACTORS:
        for policy in POLICIES:
            group = eligible[eligible["factor"].eq(factor)]
            if policy == "high_only":
                group = group[group["regime"].eq("high")]
            row = dict(factor=factor, policy=policy, **direction_metrics(group))
            day_rows = []
            for day, values in group.groupby("date"):
                day_rows.append(dict(date=day, factor=factor, policy=policy, **direction_metrics(values)))
            daily.extend(day_rows)
            for metric in ("spearman_ic", "mean_signed_return_bps", "mean_delayed_signed_return_bps"):
                values = [entry[metric] for entry in day_rows]
                row["daily_mean_" + metric] = mean_or_none(values)
                row[metric + "_ci_low"], row[metric + "_ci_high"] = daily_ci(values)
                row[metric + "_positive_days"] = sum(value is not None and value > 0 for value in values)
                row[metric + "_valid_days"] = sum(value is not None for value in values)
            summaries.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(daily)


def simulate_single_day(signals, books, settings, policy="all", entry_gate=None, exit_scheduler=None):
    """Each factor/filter policy owns its own position state and full candidate ledger."""
    if policy not in POLICIES:
        raise ValueError("unknown filter policy")
    if signals["factor"].nunique() > 1:
        raise ValueError("simulate one factor at a time")
    rows, occupied_until = [], None
    for signal in signals[signals["scheduled"]].sort_values("timestamp").to_dict("records"):
        row = dict(signal, policy=policy, status=signal["signal_status"], reason="", policy_eligible=False)
        rows.append(row)
        if signal["signal_status"] != "signal":
            continue
        if policy == "high_only" and signal["regime"] != "high":
            row.update(status="filtered_out", reason="volatility is not high")
            continue
        row["policy_eligible"] = True
        when, close = signal["timestamp"], signal["session_close"]
        if occupied_until is not None and when < occupied_until:
            row.update(status="busy", reason="previous option position remains open")
            continue
        entry_time = when + pd.Timedelta(seconds=settings.latency_seconds)
        deadline = entry_time + pd.Timedelta(seconds=settings.holding_seconds)
        if deadline > close:
            row.update(status="session_end", reason="holding period exceeds session")
            continue
        right = "CALL" if signal["direction"] == 1 else "PUT"
        contract = books.select_leg(when, signal["spot"], right, settings.max_atm_distance)
        if contract is None:
            row.update(status="missing_contract", reason="no observed ATM contract of requested right")
            continue
        strike, symbol = contract
        row.update(right=right, strike=strike, symbol=symbol, entry_time=entry_time, exit_due=deadline)
        if entry_gate is not None:
            allowed, gate_reason, details = entry_gate(signal, symbol, entry_time)
            row.update(details)
            if not allowed:
                row.update(status="entry_rejected", reason=gate_reason)
                continue
        entry, reason = books.quote(symbol, entry_time, settings.max_quote_age_seconds)
        if entry is None:
            row.update(status="entry_unavailable", reason=reason)
            continue
        if entry["mid"] < settings.min_entry_mid:
            row.update(status="entry_unavailable", reason="entry mid below configured minimum")
            continue
        slip = settings.slippage_bps / 10000
        entry_fill = entry["ask"] * (1 + slip)
        row.update(entry_mid=entry["mid"], entry_ask=entry["ask"], entry_fill=entry_fill,
                   entry_quote_age=entry["age"], entry_fee=settings.commission_per_contract,
                   entry_debit=entry_fill * settings.multiplier + settings.commission_per_contract)
        if exit_scheduler is not None:
            planned, details = exit_scheduler(signal, symbol, entry_time, deadline)
            if planned <= entry_time or planned > deadline:
                raise ValueError("scheduled exit must follow entry and not exceed original deadline")
            row.update(details)
            row.update(maximum_exit_due=deadline, exit_due=planned)
            deadline = planned
        limit = min(close, deadline + pd.Timedelta(seconds=settings.max_exit_delay_seconds))
        exit_quote, missed = None, 0
        for exit_time in pd.date_range(deadline, limit, freq="1s"):
            exit_quote, reason = books.quote(symbol, exit_time, settings.max_quote_age_seconds)
            if exit_quote is not None:
                break
            missed += 1
        row["missing_exit_checks"] = missed
        if exit_quote is None:
            row.update(status="unclosed", reason=reason, last_exit_check=limit)
            occupied_until = close
            continue
        exit_fill = exit_quote["bid"] * (1 - slip)
        mid_pnl = (exit_quote["mid"] - entry["mid"]) * settings.multiplier
        spread_cost = (entry["ask"] - entry["mid"] + exit_quote["mid"] - exit_quote["bid"]) * settings.multiplier
        slippage_cost = (entry_fill - entry["ask"] + exit_quote["bid"] - exit_fill) * settings.multiplier
        commission = 2 * settings.commission_per_contract
        net = mid_pnl - spread_cost - slippage_cost - commission
        row.update(status="closed", exit_time=exit_time, exit_mid=exit_quote["mid"],
                   exit_bid=exit_quote["bid"], exit_fill=exit_fill, exit_quote_age=exit_quote["age"],
                   exit_delay_seconds=(exit_time - deadline).total_seconds(),
                   holding_seconds=(exit_time - entry_time).total_seconds(),
                   mid_pnl=mid_pnl, spread_cost=spread_cost, slippage_cost=slippage_cost,
                   commission=commission, net_pnl=net, net_return=net / row["entry_debit"])
        occupied_until = exit_time
    return pd.DataFrame(rows)


def execution_metrics(group):
    closed = group[group["status"].eq("closed")]
    unresolved = group[group["status"].eq("unclosed")]
    realized = float(closed["net_pnl"].sum()) if len(closed) else 0.0
    debit = float(unresolved["entry_debit"].sum()) if len(unresolved) else 0.0
    result = {"candidates": len(group), "policy_signals": int(group["policy_eligible"].sum()),
              "closed": len(closed), "unclosed": len(unresolved), "unclosed_debit": debit,
              "realized_net_pnl": realized, "net_with_unclosed_zero_recovery": realized - debit,
              "delayed_exits": int(closed["exit_delay_seconds"].gt(0).sum()) if len(closed) else 0,
              "win_rate": float(closed["net_pnl"].gt(0).mean()) if len(closed) else None}
    for state in ("entry_unavailable", "missing_contract", "busy", "session_end", "filtered_out", "no_signal"):
        result[state] = int(group["status"].eq(state).sum())
    for metric in ("mid_pnl", "spread_cost", "slippage_cost", "commission", "net_pnl", "net_return"):
        result["mean_" + metric] = mean_or_none(closed[metric]) if len(closed) else None
    return result


def summarize_execution(ledger, evaluation_dates):
    summaries, daily = [], []
    evaluation = ledger[ledger["date"].isin(evaluation_dates)]
    for factor in FACTORS:
        for policy in POLICIES:
            group = evaluation[evaluation["factor"].eq(factor) & evaluation["policy"].eq(policy)]
            row = dict(factor=factor, policy=policy, **execution_metrics(group))
            day_rows = [dict(date=day, factor=factor, policy=policy,
                             **execution_metrics(group[group["date"].eq(day)])) for day in evaluation_dates]
            daily.extend(day_rows)
            means = [item["mean_net_pnl"] for item in day_rows]
            row["mean_daily_trade_pnl"] = mean_or_none(means)
            row["mean_daily_trade_pnl_ci_low"], row["mean_daily_trade_pnl_ci_high"] = daily_ci(means)
            row["positive_realized_days"] = sum(item["realized_net_pnl"] > 0 for item in day_rows)
            row["positive_stress_days"] = sum(item["net_with_unclosed_zero_recovery"] > 0 for item in day_rows)
            row["evaluation_days"] = len(day_rows)
            summaries.append(row)
    daily = pd.DataFrame(daily)
    comparisons = []
    for factor in FACTORS:
        group = daily[daily["factor"].eq(factor)]
        for metric in ("mean_net_pnl", "realized_net_pnl", "net_with_unclosed_zero_recovery"):
            paired = group.pivot(index="date", columns="policy", values=metric)
            diff = (paired["high_only"] - paired["all"]).dropna()
            low, high = daily_ci(diff)
            comparisons.append({"factor": factor, "metric": metric, "paired_days": len(diff),
                                "high_only_minus_all": mean_or_none(diff), "ci_low": low, "ci_high": high,
                                "positive_days": int(diff.gt(0).sum())})
    return pd.DataFrame(summaries), daily, pd.DataFrame(comparisons)


def markdown_table(frame, columns, labels, percent=()):
    rows = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.to_dict("records"):
        cells = []
        for name in columns:
            value = record[name]
            if value is None or pd.isna(value):
                cell = "—"
            elif name == "factor":
                cell = FACTORS[value]
            elif name == "policy":
                cell = "全部时段" if value == "all" else "仅高波动"
            elif name in percent:
                cell = "{:.2%}".format(value)
            elif isinstance(value, (float, np.floating)):
                cell = "{:.4f}".format(value)
            else:
                cell = str(value)
            cells.append(cell)
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def filter_sensitivity(daily):
    """Omit each date in turn as a diagnostic; never alter the main experiment."""
    rows = []
    for factor, group in daily.groupby("factor", sort=True):
        paired = group.pivot(index="date", columns="policy", values="mean_net_pnl")
        differences = (paired["high_only"] - paired["all"]).dropna()
        for omitted in differences.index:
            remaining = differences.drop(omitted)
            low, high = daily_ci(remaining)
            rows.append({"factor": factor, "omitted_date": omitted, "paired_days": len(remaining),
                         "high_only_minus_all": mean_or_none(remaining), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def write_report(output, settings, direction, execution, filter_comparison, metadata):
    lines = ["# 三个盘口因子：方向、单腿期权成本与高波动过滤", "",
             "前 {} 天初始化，后 {} 天历史滚动验证。全部日期已参与此前研究，尚无新增独立样本。".format(
                 settings.calibration_days, len(metadata["evaluation_dates"])), "",
             "## 1. 方向关系", "",
             "因子固定为 ofi_5s、buy_volume_5s - sell_volume_5s、depth_imbalance。前五个交易日相同半小时段确定 25%/75% 分位。",
             "因子大于 max(历史75%分位, 0) 看涨，小于 min(历史25%分位, 0) 看跌，其他不交易。严格不等号保留零值与重复值，不按结果反转方向。", "",
             "以下命中率只针对强信号；无价格变化也计入分母。随机方向基线为同一批有效标签的非零涨跌占比的一半。IC 为全部合格观测的每日 Spearman 相关均值。", ""]
    lines += markdown_table(direction[direction["policy"].eq("all")],
                            ["factor", "signals", "labelled_signals", "hit_rate", "random_sign_hit_baseline",
                             "daily_mean_spearman_ic", "mean_signed_return_bps", "mean_delayed_signed_return_bps"],
                            ["因子", "强信号", "有效30秒标签", "方向命中率", "随机方向基线", "每日IC均值",
                             "预测方向收益/bps", "延迟后方向收益/bps"],
                            percent=("hit_rate", "random_sign_hit_baseline"))
    lines += ["", "下表用每日等权均值检查方向收益的稳定性，区间按交易日重抽样；其权重不同于上表的逐信号均值。", ""]
    lines += markdown_table(direction,
                            ["factor", "policy", "daily_mean_mean_signed_return_bps",
                             "mean_signed_return_bps_ci_low", "mean_signed_return_bps_ci_high",
                             "mean_signed_return_bps_positive_days", "mean_signed_return_bps_valid_days"],
                            ["因子", "时段", "每日方向收益均值/bps", "95%下界", "95%上界", "正收益天数", "有效天数"])
    lines += ["", "方向收益为 预测符号 × QQQ中价收益，并非成交利润。普通标签取 t→t+30s；延迟标签取 t+1s→t+31s，边界、延迟和持有期随配置变动。",
              "future_outcomes 仅用于评估，不传入阈值或成交模拟。缺失标签单独计数，不影响是否发出交易信号。", "",
              "## 2. 单腿期权成本", "",
              "正信号买一张 CALL，负信号买一张 PUT；信号时选择已观测到的最近执行价，整笔保持同一合约。每 {} 秒评估，延迟 {} 秒入场，持有 {} 秒后退出，退出最多等待 {} 秒。".format(
                  settings.signal_seconds, settings.latency_seconds, settings.holding_seconds, settings.max_exit_delay_seconds),
              "按 ask 买入、bid 卖出，每次每张手续费 ${:.2f}，买/卖价格分别加/减 {} bps；乘数 {}。每个因子与过滤方案有独立持仓状态，同一时刻最多一张期权。".format(
                  settings.commission_per_contract, settings.slippage_bps, settings.multiplier), "",
              "下表金额单位为美元/张，平均损益只统计完成退出的交易。", ""]
    lines += markdown_table(execution,
                            ["factor", "policy", "closed", "mean_mid_pnl", "mean_spread_cost", "mean_slippage_cost",
                             "mean_commission", "mean_net_pnl", "win_rate", "positive_realized_days"],
                            ["因子", "时段", "已退出", "中价毛损益", "价差成本", "滑点", "手续费", "净损益",
                             "净盈利率", "正已实现收益天数"], percent=("win_rate",))
    lines += ["", "## 3. 高波动过滤", "",
              "仅高波动方案复用完全相同的因子阈值，只在过去30秒波动高于前序交易日同时间段75%分位时入场。",
              "全部时段与仅高波动分别重跑持仓，不能用全部时段已成交记录的子集替代过滤方案。两方案共享数据但不是可叠加的独立收益。",
              "下表比较每日平均每笔净损益，按交易日配对重抽样 10,000 次；高波动交易次数下降会减少总成本，不能仅凭累计亏损变小判定改善。", ""]
    lines += markdown_table(filter_comparison[filter_comparison["metric"].eq("mean_net_pnl")],
                            ["factor", "paired_days", "high_only_minus_all", "ci_low", "ci_high", "positive_days"],
                            ["因子", "配对天数", "高波动减全部/美元", "95%下界", "95%上界", "差额为正天数"])
    lines += ["", "每日等权统计会给交易较少的日期相同权重。filter_leave_one_day_out.csv 逐一省略每个日期检查敏感性；这只是诊断，完整结果不删除任何日期。"]
    lines += ["", "## 执行审计", ""]
    lines += markdown_table(execution,
                            ["factor", "policy", "policy_signals", "closed", "entry_unavailable", "missing_contract",
                             "busy", "session_end", "unclosed", "unclosed_debit", "net_with_unclosed_zero_recovery"],
                            ["因子", "时段", "入选信号", "已退出", "入场报价缺失", "无合约", "持仓占用", "临近收盘",
                             "超时未退出", "未退出入场资金", "零回收压力净损益"])
    lines += ["", "- 未退出交易保留已支付资金，且当日不再入场。已实现损益不含这些头寸；零回收压力值不是实际到期结算。",
              "- 报价来自完整 normalized 合约簿，按照 available_at 向后查询；最新无效更新不被更早有效报价替代。未要求另一侧期权也存在。",
              "- 入场中价至少 ${}，退出不设此门槛；报价年龄上限 {} 秒，双边数量都至少一张。未模拟排队、部分成交和市场冲击。".format(settings.min_entry_mid, settings.max_quote_age_seconds),
              "- 主动买卖量根据源数据 TradeDirection.Up/Down 分类，属于源数据提供的方向代理。OFI 是一秒快照累计量。",
              "- 交易日按纽约 09:30–16:00 固定时段处理，未支持半日历。输入来自已有特征缓存，内容校验值在 study.json。",
              "- 仅检查三个预先固定的同向因子，不调整窗口或根据测试表现反向。区间是探索性描述，未校正多个比较，不能代替新增日期的验证。",
              "- 方向统计不受期权未来是否能成交影响；期权已退出收益仍受报价可用性和仓位占用影响，应结合全部候选审计阅读。", "",
              "## 复现", "", "```bash",
              'python -m research_engine.analysis.directional_factors --data-root "$SF_CLOUD/data"', "```", "",
              "direction_comparison.csv / direction_daily.csv 为方向统计；execution_comparison.csv / execution_daily.csv 为成本后结果；",
              "filter_comparison.csv 为配对差异，filter_leave_one_day_out.csv 为逐日省略诊断；signals.parquet 保存纯因果信号，direction_observations.parquet 另存未来评估标签；",
              "opportunities.csv 与 execution_audit.csv 记录各策略的全部候选和异常成交；study.json 保存协议、环境与数据/代码校验值。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_study(data_root, output, settings):
    paths = sorted((data_root / "features").glob("*.parquet"))
    if len(paths) <= settings.calibration_days:
        raise ValueError("need more dates than calibration_days")
    evaluation_dates = [path.stem for path in paths[settings.calibration_days:]]
    for day in evaluation_dates:
        if not (data_root / "normalized" / day / "events.parquet").exists():
            raise ValueError("missing normalized contract books for " + day)
    history, signal_frames, label_frames, manifest = [], [], [], []
    for path in paths:
        frame = pd.read_parquet(path, columns=INPUT_COLUMNS)
        observed = factor_observations(frame, settings)
        if observed.empty or set(observed["date"]) != {path.stem}:
            raise ValueError("feature filename/date mismatch")
        signals = factor_signals(observed, history, settings)
        signal_frames.append(signals)
        label_frames.append(signals.merge(future_outcomes(frame, settings), on="timestamp", how="left", validate="many_to_one"))
        history.append(observed)
        manifest.append({"file": "features/" + path.name, "sha256": file_hash(path)})
    signals = pd.concat(signal_frames, ignore_index=True)
    panel = pd.concat(label_frames, ignore_index=True)
    direction, direction_daily = summarize_direction(panel)
    output.mkdir(parents=True, exist_ok=False)
    signals.to_parquet(output / "signals.parquet", index=False)
    panel.to_parquet(output / "direction_observations.parquet", index=False)
    direction.to_csv(output / "direction_comparison.csv", index=False)
    direction_daily.to_csv(output / "direction_daily.csv", index=False)
    print("Direction assessment complete; starting independent option policies.", flush=True)
    ledgers = []
    for day, day_signals in signals.groupby("date", sort=True):
        if day in evaluation_dates:
            path = data_root / "normalized" / day / "events.parquet"
            books = read_books(path, day)
            manifest.append({"file": "normalized/" + day + "/events.parquet", "sha256": file_hash(path)})
        else:
            books = None  # warmup never accesses a contract book
        counts = {}
        for factor in FACTORS:
            selected = day_signals[day_signals["factor"].eq(factor)]
            for policy in POLICIES:
                ledger = simulate_single_day(selected, books, settings, policy)
                ledgers.append(ledger)
                counts[factor + "/" + policy] = ledger["status"].value_counts().to_dict()
        print(json.dumps({"date": day, "status_counts": counts}), flush=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    execution, execution_daily, filtering = summarize_execution(ledger, evaluation_dates)
    ledger.to_csv(output / "opportunities.csv", index=False)
    execution.to_csv(output / "execution_comparison.csv", index=False)
    execution_daily.to_csv(output / "execution_daily.csv", index=False)
    filtering.to_csv(output / "filter_comparison.csv", index=False)
    filter_sensitivity(execution_daily).to_csv(output / "filter_leave_one_day_out.csv", index=False)
    audit = (~ledger["status"].isin(["closed", "warmup", "no_signal", "filtered_out"]) |
             ledger.get("exit_delay_seconds", pd.Series(0, index=ledger.index)).gt(0))
    ledger[audit].to_csv(output / "execution_audit.csv", index=False)
    package = Path(__file__).resolve().parents[1]
    metadata = {"study": "three_fixed_factors_single_options_and_volatility_filter",
                "validation": "retrospective_walk_forward_previously_explored_dates",
                "settings": asdict(settings), "factors": FACTORS,
                "rule": "positive_above_prior_q75_CALL_negative_below_prior_q25_PUT",
                "dates": [path.stem for path in paths], "evaluation_dates": evaluation_dates,
                "source_manifest": manifest,
                "code_manifest": [{"file": str(path.relative_to(package.parent)), "sha256": file_hash(path)}
                                  for path in sorted(package.rglob("*.py"))],
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "pandas": pd.__version__, "pyarrow": version("pyarrow")}}
    (output / "study.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output, settings, direction, execution, filtering, metadata)
    return execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    root = args.output_root or resolve_path(load_config(), "results")
    output = root.expanduser() / ("directional_factors_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    results = run_study(args.data_root.expanduser(), output, StudySettings())
    print(results[["factor", "policy", "closed", "mean_net_pnl", "unclosed"]].to_string(index=False))
    print("Report: " + output.name)


if __name__ == "__main__":
    main()
