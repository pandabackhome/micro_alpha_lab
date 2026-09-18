"""Fixed pressure events, option-response gaps, and option-flow confirmation."""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import (
    FEATURE_COLUMNS, GROUPS, StudySettings, assign_regimes, observations, read_books, file_hash,
)
from research_engine.analysis.directional_factors import (
    daily_ci, execution_metrics, future_outcomes, mean_or_none, simulate_single_day,
)
from research_engine.config import load_config, resolve_path


RULES = {
    "pressure_only": "强压力对照",
    "pressure_breakout": "压力增强且突破",
    "pressure_exhaustion": "压力衰减且价格未响应",
    "spot_impulse": "正股强变动对照",
    "option_response_gap": "期权剩余响应超过成本",
    "breakout_confirmed": "突破加期权成交确认",
    "exhaustion_confirmed": "衰减加期权成交确认",
}
COMPARISONS = {
    "pressure_breakout": "pressure_only", "pressure_exhaustion": "pressure_only",
    "option_response_gap": "spot_impulse", "breakout_confirmed": "pressure_breakout",
    "exhaustion_confirmed": "pressure_exhaustion",
}
HORIZONS = (30, 180)
STRICT_AGE = 1.0
INPUT_COLUMNS = FEATURE_COLUMNS + ["bid_size", "ask_size", "buy_volume_5s", "sell_volume_5s",
                                  "call_buy_volume", "call_sell_volume", "put_buy_volume", "put_sell_volume"]


def observed_events(frame, settings):
    """Causal event features on the same validated, complete one-second grid."""
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    base = observations(frame, settings)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns")
    bid, ask, age = frame.qqq_bid, frame.qqq_ask, frame.qqq_depth_age_seconds
    good = age.between(0, settings.max_quote_age_seconds) & bid.gt(0) & ask.ge(bid) & np.isfinite(bid) & np.isfinite(ask)
    mid = ((bid + ask) / 2).where(good)
    depth_good = good & frame.bid_size.gt(0) & frame.ask_size.gt(0) & np.isfinite(frame.bid_size) & np.isfinite(frame.ask_size)
    depth = (frame.bid_size + frame.ask_size).where(depth_good).rolling(5, min_periods=5).mean()
    volume_good = frame.buy_volume_5s.ge(0) & frame.sell_volume_5s.ge(0)
    pressure = ((frame.buy_volume_5s - frame.sell_volume_5s) / depth).where(volume_good)
    fields = pd.DataFrame({
        "timestamp": frame.timestamp, "pressure": pressure, "previous_pressure": pressure.shift(5),
        "spot_change_5s": mid - mid.shift(5), "spot_before_5s": mid.shift(5),
        "spot_age": age, "spot_before_age": age.shift(5), "spot_spread": ask - bid,
        "previous_high": mid.shift(1).rolling(30, min_periods=30).max(),
        "previous_low": mid.shift(1).rolling(30, min_periods=30).min(),
    })
    cols = ["call_buy_volume", "call_sell_volume", "put_buy_volume", "put_sell_volume"]
    flow = frame[cols]
    total = flow.sum(axis=1)
    valid_flow = flow.ge(0).all(axis=1) & np.isfinite(flow).all(axis=1) & total.gt(0)
    fields["option_flow"] = ((flow.call_buy_volume - flow.call_sell_volume - flow.put_buy_volume + flow.put_sell_volume) / total).where(valid_flow)
    return base.merge(fields, on="timestamp", validate="one_to_one")


def option_observations(observed, books, settings):
    """Same-symbol past changes only; quote failures remain visible as reasons."""
    rows = []
    for record in observed.to_dict("records"):
        row = dict(record)
        when = record["timestamp"]
        before = when - pd.Timedelta(seconds=5)
        spot_ok = (pd.notna(record["past_vol"]) and np.isfinite(record["spot_change_5s"]) and
                   0 <= record["spot_age"] <= STRICT_AGE and 0 <= record["spot_before_age"] <= STRICT_AGE)
        for right in ("CALL", "PUT"):
            prefix = right.lower() + "_"
            row.update({prefix + "symbol": None, prefix + "change_5s": np.nan,
                        prefix + "mid_before": np.nan, prefix + "mid_now": np.nan,
                        prefix + "spread": np.nan, prefix + "cost_estimate": np.nan,
                        prefix + "quote_age": np.nan, prefix + "quote_before_age": np.nan,
                        prefix + "response_status": "invalid_spot"})
            if not spot_ok:
                continue
            contract = books.select_leg(when, record["spot"], right, settings.max_atm_distance)
            if contract is None:
                row[prefix + "response_status"] = "missing_contract"
                continue
            row[prefix + "symbol"] = contract[1]
            current, current_reason = books.quote(contract[1], when, STRICT_AGE)
            old, old_reason = books.quote(contract[1], before, STRICT_AGE)
            if current is None or old is None:
                row[prefix + "response_status"] = "now:" + current_reason + ";past:" + old_reason
                continue
            if when - pd.Timedelta(seconds=current["age"]) <= before:
                row[prefix + "response_status"] = "not_refreshed"
                continue
            row.update({prefix + "change_5s": current["mid"] - old["mid"],
                        prefix + "mid_before": old["mid"], prefix + "mid_now": current["mid"],
                        prefix + "spread": current["ask"] - current["bid"],
                        prefix + "cost_estimate": estimated_cost(current, settings),
                        prefix + "quote_age": current["age"], prefix + "quote_before_age": old["age"],
                        prefix + "response_status": "ok"})
        rows.append(row)
    return pd.DataFrame(rows)


def estimated_cost(quote, settings):
    return (quote["ask"] - quote["bid"] + 2 * settings.commission_per_contract / settings.multiplier +
            (quote["ask"] + quote["bid"]) * settings.slippage_bps / 10000)


def calibrate(current, history, settings):
    result = assign_regimes(current, history, settings)
    for name in ("pressure_q75", "impulse_q75", "flow_q75", "call_beta", "put_beta"):
        result[name] = np.nan
    for name in ("pressure_samples", "impulse_samples", "flow_samples", "call_beta_samples", "put_beta_samples"):
        result[name] = 0
    if len(history) < settings.calibration_days:
        return result
    past = pd.concat(history[-settings.calibration_days:], ignore_index=True)
    for bucket, indices in result.groupby("half_hour").groups.items():
        group = past[past.half_hour.eq(bucket) & past.past_vol.notna()]
        for field, prefix in (("pressure", "pressure"), ("spot_change_5s", "impulse"), ("option_flow", "flow")):
            values = group[field].replace([np.inf, -np.inf], np.nan).dropna().abs()
            result.loc[indices, prefix + "_samples"] = len(values)
            if len(values) >= settings.min_history_samples:
                result.loc[indices, prefix + "_q75"] = values.quantile(.75)
        for right in ("call", "put"):
            pair = group[group[right + "_response_status"].eq("ok")]
            x, y = pair.spot_change_5s, pair[right + "_change_5s"]
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            result.loc[indices, right + "_beta_samples"] = len(x)
            if len(x) < settings.min_history_samples or x.ne(0).sum() < 20:
                continue
            beta = float((x * y).sum() / x.pow(2).sum())
            if (right == "call" and 0 < beta <= 1) or (right == "put" and -1 <= beta < 0):
                result.loc[indices, right + "_beta"] = beta
    return result


def make_signals(calibrated):
    rows = []
    for record in calibrated.to_dict("records"):
        p, old = record["pressure"], record["previous_pressure"]
        q = record["pressure_q75"]
        pressure_ok = np.isfinite([p, old, record["previous_high"], record["previous_low"]]).all()
        pressure_ready = pressure_ok and np.isfinite(q)
        strong = pressure_ready and abs(p) > q
        prior_strong = pressure_ready and abs(old) > q
        breakout = prior_strong and p * old > 0 and abs(p) >= abs(old) and (
            (p > 0 and record["spot"] > record["previous_high"]) or
            (p < 0 and record["spot"] < record["previous_low"]))
        exhaustion = (prior_strong and abs(p) <= .5 * abs(old) and
                      abs(record["spot_change_5s"]) <= record["spot_spread"])
        base_dirs = {"pressure_only": int(np.sign(p)) if strong else 0,
                     "pressure_breakout": int(np.sign(p)) if breakout else 0,
                     "pressure_exhaustion": -int(np.sign(old)) if exhaustion else 0}
        move = record["spot_change_5s"]
        impulse_direction = int(np.sign(move)) if np.isfinite(move) else 0
        side = "call" if impulse_direction > 0 else "put"
        beta, cost = record[side + "_beta"], record[side + "_cost_estimate"]
        gap = beta * move - record[side + "_change_5s"]
        impulse = np.isfinite(record["impulse_q75"]) and abs(move) > record["impulse_q75"]
        response_ok = record[side + "_response_status"] == "ok" and np.isfinite(beta)
        for rule in RULES:
            row = dict(record, factor=rule, direction=0, signal_status="no_signal")
            rows.append(row)
            if record["regime"] not in GROUPS:
                row["signal_status"] = record["regime"]
                continue
            if rule in ("spot_impulse", "option_response_gap"):
                row.update(response_side=side, response_gap=gap, response_cost=cost,
                           response_symbol=record[side + "_symbol"], response_beta=beta)
                if not np.isfinite(record["impulse_q75"]):
                    row["signal_status"] = "insufficient_impulse_history"
                elif not response_ok:
                    row["signal_status"] = "response_unavailable"
                elif impulse and (rule == "spot_impulse" or gap > cost):
                    row.update(direction=impulse_direction, signal_status="signal")
                continue
            if not pressure_ok:
                row["signal_status"] = "invalid_pressure"
                continue
            if not pressure_ready:
                row["signal_status"] = "insufficient_pressure_history"
                continue
            original = {"breakout_confirmed": "pressure_breakout", "exhaustion_confirmed": "pressure_exhaustion"}.get(rule, rule)
            direction = base_dirs[original]
            if rule.endswith("_confirmed"):
                flow, threshold = record["option_flow"], record["flow_q75"]
                if not np.isfinite([flow, threshold]).all():
                    row["signal_status"] = "confirmation_unavailable"
                    continue
                if direction and not direction * flow > threshold:
                    row["signal_status"] = "unconfirmed"
                    continue
            if direction:
                row.update(direction=direction, signal_status="signal")
    return pd.DataFrame(rows)


def latency_diagnostics(signals, frame, books, settings):
    """Future checks are written separately and cannot alter the decision panel."""
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).as_unit("ns")
    chosen = signals[signals.factor.eq("option_response_gap") & signals.scheduled & signals.signal_status.eq("signal")]
    rows = []
    for signal in chosen.to_dict("records"):
        entry_time = signal["timestamp"] + pd.Timedelta(seconds=settings.latency_seconds)
        row = {key: signal[key] for key in ("date", "timestamp", "direction", "response_gap", "response_cost", "response_symbol")}
        row.update(entry_time=entry_time, status="unavailable", gap_at_entry=np.nan, cost_at_entry=np.nan)
        quote, reason = books.quote(signal["response_symbol"], entry_time, STRICT_AGE)
        row["reason"] = reason
        if quote is not None and entry_time in frame.index:
            spot = frame.loc[entry_time]
            if (0 <= spot.qqq_depth_age_seconds <= STRICT_AGE and
                    np.isfinite([spot.qqq_bid, spot.qqq_ask]).all() and 0 < spot.qqq_bid <= spot.qqq_ask):
                move = (spot.qqq_bid + spot.qqq_ask) / 2 - signal["spot_before_5s"]
                gap = signal["response_beta"] * move - (quote["mid"] - signal[signal["response_side"] + "_mid_before"])
                cost = estimated_cost(quote, settings)
                row.update(status="persists" if gap > cost else "disappears", gap_at_entry=gap, cost_at_entry=cost,
                           entry_quote_age=quote["age"], reason="ok")
            else:
                row["reason"] = "invalid_spot_at_entry"
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", "timestamp", "direction", "response_gap", "response_cost", "response_symbol",
                                       "entry_time", "status", "gap_at_entry", "cost_at_entry", "reason", "entry_quote_age"])


def direction_metrics(group):
    selected = group[group.signal_status.eq("signal") & group.scheduled]
    valid = selected[selected.future_return_bps.notna()]
    delayed = selected[selected.delayed_return_bps.notna()]
    signed = valid.direction * valid.future_return_bps
    delayed_signed = delayed.direction * delayed.delayed_return_bps
    return {"signals": len(selected), "labelled": len(valid), "missing_labels": len(selected) - len(valid),
            "hit_rate": mean_or_none(signed.gt(0)), "random_sign_baseline": mean_or_none(valid.future_return_bps.ne(0) * .5),
            "mean_signed_bps": mean_or_none(signed), "delayed_labelled": len(delayed),
            "mean_delayed_signed_bps": mean_or_none(delayed_signed)}


def summarize(panel, ledger, evaluation_dates):
    execution, direction, execution_daily, direction_daily = [], [], [], []
    selected = ledger[ledger.date.isin(evaluation_dates)]
    labels = panel[panel.date.isin(evaluation_dates)]
    for rule in RULES:
        for horizon in HORIZONS:
            eg = selected[selected.factor.eq(rule) & selected.horizon.eq(horizon)]
            dg = labels[labels.factor.eq(rule) & labels.horizon.eq(horizon)]
            er = dict(factor=rule, horizon=horizon, **execution_metrics(eg))
            dr = dict(factor=rule, horizon=horizon, **direction_metrics(dg))
            ed = [dict(date=day, factor=rule, horizon=horizon, **execution_metrics(eg[eg.date.eq(day)])) for day in evaluation_dates]
            dd = [dict(date=day, factor=rule, horizon=horizon, **direction_metrics(dg[dg.date.eq(day)])) for day in evaluation_dates]
            execution_daily.extend(ed)
            direction_daily.extend(dd)
            for row, days, metric in ((er, ed, "mean_net_pnl"), (dr, dd, "mean_signed_bps"), (dr, dd, "mean_delayed_signed_bps")):
                values = [item[metric] for item in days]
                row["daily_" + metric] = mean_or_none(values)
                row[metric + "_ci_low"], row[metric + "_ci_high"] = daily_ci(values)
                row[metric + "_valid_days"] = sum(value is not None for value in values)
            er["positive_realized_days"] = sum(item["realized_net_pnl"] > 0 for item in ed)
            er["positive_stress_days"] = sum(item["net_with_unclosed_zero_recovery"] > 0 for item in ed)
            execution.append(er)
            direction.append(dr)
    return tuple(pd.DataFrame(rows) for rows in (execution, direction, execution_daily, direction_daily))


def paired_comparisons(daily):
    comparisons, sensitivity = [], []
    for horizon in HORIZONS:
        for metric in ("mean_net_pnl", "realized_net_pnl", "net_with_unclosed_zero_recovery"):
            pivot = daily[daily.horizon.eq(horizon)].pivot(index="date", columns="factor", values=metric)
            for rule, control in COMPARISONS.items():
                diff = (pivot[rule] - pivot[control]).dropna()
                low, high = daily_ci(diff)
                comparisons.append({"factor": rule, "control": control, "horizon": horizon, "metric": metric,
                                    "paired_days": len(diff), "mean_difference": mean_or_none(diff), "ci_low": low, "ci_high": high})
                if metric != "mean_net_pnl":
                    continue
                for omitted in diff.index:
                    remaining = diff.drop(omitted)
                    lo, hi = daily_ci(remaining)
                    sensitivity.append({"factor": rule, "control": control, "horizon": horizon, "omitted_date": omitted,
                                        "paired_days": len(remaining), "mean_difference": mean_or_none(remaining),
                                        "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(comparisons), pd.DataFrame(sensitivity)


def profit_sensitivity(daily):
    """Preserve every date in the main result; diagnose profit concentration."""
    rows = []
    for (rule, horizon), group in daily.groupby(["factor", "horizon"]):
        for omitted in group.date:
            remaining = group[~group.date.eq(omitted)]
            count = int(remaining.closed.sum())
            net = float(remaining.realized_net_pnl.sum())
            rows.append({"factor": rule, "horizon": horizon, "omitted_date": omitted,
                         "closed": count, "realized_net_pnl": net, "mean_net_pnl": net / count if count else None,
                         "unclosed_debit": float(remaining.unclosed_debit.sum()),
                         "net_with_unclosed_zero_recovery": float(remaining.net_with_unclosed_zero_recovery.sum())})
    return pd.DataFrame(rows)


def latency_attribution(ledger, latency):
    """Descriptive attribution of original fills, not a newly filtered strategy."""
    chosen = ledger[ledger.factor.eq("option_response_gap") & ledger.status.eq("closed")]
    if chosen.empty:
        return pd.DataFrame(columns=["horizon", "latency_status", "count", "mean", "sum"])
    joined = chosen.merge(latency[["timestamp", "status"]].rename(columns={"status": "latency_status"}),
                          on="timestamp", how="left", validate="many_to_one")
    return joined.groupby(["horizon", "latency_status"]).net_pnl.agg(["count", "mean", "sum"]).reset_index()


def table(frame, columns, labels, percentages=()):
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict("records"):
        cells = []
        for col in columns:
            value = row[col]
            if value is None or pd.isna(value):
                text = "—"
            elif col in ("factor", "control"):
                text = RULES[value]
            elif col in percentages:
                text = "{:.2%}".format(value)
            elif isinstance(value, (float, np.floating)):
                text = "{:.4f}".format(value)
            else:
                text = str(value)
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(output, execution, direction, comparison, latency, attribution, concentration, metadata):
    lines = ["# 压力事件、期权响应与成交确认", "",
             "{} 个历史日期，前 {} 天校准，后 {} 天评估。此前已探索这些日期；所有结果均为历史探索。".format(
                 len(metadata["dates"]), metadata["settings"]["calibration_days"], len(metadata["evaluation_dates"])),
             "同目录 protocol.md 为运行前固定协议。每 60 秒检查事件，每 30 秒取历史校准观测；30 秒、180 秒独立执行。", "",
             "## 1. 压力事件", "",
             "压力为 5 秒买卖量差除以 5 秒平均最优双边盘口量。强压力阈值来自前五天同一半小时。",
             "突破：旧压力强、当前同号且不减、价格突破过去30秒区间。衰减：旧压力强、当前绝对值降至一半以内、5秒价格变化不超过当前价差，反向交易。", ""]
    first = execution[execution.factor.isin(["pressure_only", "pressure_breakout", "pressure_exhaustion"])]
    columns = ["factor", "horizon", "policy_signals", "closed", "mean_mid_pnl", "mean_spread_cost", "mean_slippage_cost", "mean_commission", "mean_net_pnl", "positive_realized_days"]
    headers = ["规则", "持有/秒", "信号", "已退出", "中价毛损益", "价差成本", "滑点", "手续费", "净损益", "正已实现天数"]
    lines += table(first, columns, headers)
    lines += ["", "金额均为美元/张；均值只含已经退出的交易，未退出资金在审计表单列。", "",
              "## 2. 期权响应滞后", "",
              "两组都要求正股5秒变动超过历史75%分位、同一合约两端与正股两端报价年龄≤1秒，以及有效的历史响应系数。",
              "剩余响应 = 历史系数 × 正股5秒变化 − 同一合约5秒中价变化；过滤组要求它超过当前价差、往返手续费与滑点估算。",
              "响应系数不是 Delta 或公允价，未解释波动率、时间价值和异步报价的全部变化；存在残差并不保证随后收敛。", ""]
    lines += table(execution[execution.factor.isin(["spot_impulse", "option_response_gap"])], columns, headers)
    lines += ["", "剩余响应组的逐日省略诊断（仅检验集中度，完整结果保留所有日期）："]
    for horizon in HORIZONS:
        group = concentration[concentration.factor.eq("option_response_gap") & concentration.horizon.eq(horizon)]
        if len(group):
            worst = group.loc[group.realized_net_pnl.idxmin()]
            lines += ["- 持有 {} 秒：省略 {} 后，剩余 {} 笔合计净损益 ${:.4f}。这是逐一省略所有日期中累计净损益最低的情况。".format(
                horizon, worst.omitted_date, worst.closed, worst.realized_net_pnl)]
    counts = latency.status.value_counts().to_dict() if len(latency) else {}
    lines += ["", "延迟诊断独立于交易选择：{} 个剩余响应信号中，1秒后仍超过新成本 {} 个，已不超过 {} 个，因报价或正股不可用无法判断 {} 个。".format(
        len(latency), counts.get("persists", 0), counts.get("disappears", 0), counts.get("unavailable", 0)),
        "这里只检查信号残差是否持续，不把它当作实际成交或利润。下面按该诊断归因原模拟已退出交易，不重新筛选持仓。persists=仍超过成本，disappears=已不超过，unavailable=严格一秒新鲜度诊断不可用（一般成交仍允许五秒内报价）。", ""]
    lines += table(attribution, ["horizon", "latency_status", "count", "mean", "sum"],
                   ["持有/秒", "一秒后状态", "已退出", "每笔净损益", "合计净损益"])
    lines += ["", "对照尚未匹配正股变动幅度、时点和期权价格，因此不能把收益差全部归因于滞后。", "", "## 3. 期权成交确认", "",
              "确认使用5秒 CALL净买量减PUT净买量、除以双边成交总量；绝对值超过历史75%分位且与交易同向。没有开平仓信息，Up/Down 是源数据方向代理。", ""]
    lines += table(execution[execution.factor.isin(["breakout_confirmed", "exhaustion_confirmed"])], columns, headers)
    lines += ["", "## 方向关系（不含期权成本）", ""]
    lines += table(direction, ["factor", "horizon", "signals", "labelled", "missing_labels", "hit_rate", "random_sign_baseline", "mean_signed_bps", "mean_delayed_signed_bps"],
                   ["规则", "期限/秒", "信号", "有效标签", "缺失", "命中率", "随机方向基线", "顺方向收益/bps", "延迟后/bps"], ("hit_rate", "random_sign_baseline"))
    lines += ["", "无涨跌也计入命中率分母。随机方向基线为非零涨跌比例的一半。未来标签单独存放，不参与决策。", "",
              "## 每日收益稳定性", ""]
    lines += table(execution, ["factor", "horizon", "daily_mean_net_pnl", "mean_net_pnl_ci_low", "mean_net_pnl_ci_high", "mean_net_pnl_valid_days"],
                   ["规则", "持有/秒", "每日每笔净损益均值", "95%下界", "95%上界", "有效日期"])
    lines += ["", "## 相对对照的变化", "",
              "以下为每日平均每笔净损益之差，按日期配对，重抽样10,000次。仅在两组当天都有已退出交易时配对；实际持仓分别模拟。", ""]
    lines += table(comparison[comparison.metric.eq("mean_net_pnl")], ["factor", "control", "horizon", "paired_days", "mean_difference", "ci_low", "ci_high"],
                   ["规则", "对照", "期限/秒", "配对天数", "差额/美元", "95%下界", "95%上界"])
    lines += ["", "日期很少，区间只是探索性描述，未校正多个规则/期限比较；逐日省略对照诊断见 leave_one_day_out.csv，总利润集中度见 profit_leave_one_day_out.csv，不从主结果删除日期。", "", "## 执行审计", ""]
    lines += table(execution, ["factor", "horizon", "policy_signals", "closed", "entry_unavailable", "missing_contract", "busy", "session_end", "unclosed", "unclosed_debit", "net_with_unclosed_zero_recovery"],
                   ["规则", "期限/秒", "信号", "已退出", "入场报价缺失", "无合约", "持仓占用", "临近收盘", "未退出", "未退出入场资金", "零回收压力总损益"])
    lines += ["", "- 正信号买CALL、负信号买PUT；延迟1秒，ask买、bid卖；单次每张$0.65，额外1bps，乘数100。",
              "- 退出报价来自完整合约簿，最多等30秒；未平仓不记为零收益，当日不再开仓。零回收是压力假设，并非实际结算。",
              "- 180秒持有可能阻止后续入场，方向标签可能重叠；按日期统计，不将每笔当成独立样本。",
              "- 各规则持仓独立，结果不能相加。未模拟排队、部分成交、市场冲击或半日交易日历。",
              "- response_unavailable 保留缺少新鲜同合约报价或历史系数的候选，不能解释为没有响应滞后。",
              "- status_counts.csv 保存所有候选状态；observations.parquet 保存包括不合格报价原因的历史响应观测。", "",
              "## 复现", "", "```bash", 'python -m research_engine.analysis.event_response --data-root "$SF_CLOUD/data"', "```", "",
              "signals.parquet 不含未来标签；direction_observations.parquet 单独存评估标签；latency_diagnostics.csv 为一秒后检查。",
              "opportunities.csv / execution_audit.csv 保留完整候选与执行异常。study.json 记录协议、输入和源码校验值。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_study(data_root, output, settings):
    paths = sorted((data_root / "features").glob("*.parquet"))
    if len(paths) <= settings.calibration_days:
        raise ValueError("need more dates than calibration_days")
    for path in paths:
        if not (data_root / "normalized" / path.stem / "events.parquet").exists():
            raise ValueError("full contract books are required on calibration and evaluation dates: " + path.stem)
    package = Path(__file__).resolve().parents[1]
    protocol = package.parent / "research" / "event_response_protocol.md"
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.md").write_text(protocol.read_text(encoding="utf-8"), encoding="utf-8")
    history, signal_frames, label_frames, ledgers, diagnostics, manifest = [], [], [], [], [], []
    for path in paths:
        frame = pd.read_parquet(path, columns=INPUT_COLUMNS)
        observed = observed_events(frame, settings)
        if observed.empty or set(observed.date) != {path.stem}:
            raise ValueError("feature filename/date mismatch")
        book_path = data_root / "normalized" / path.stem / "events.parquet"
        books = read_books(book_path, path.stem)
        current = option_observations(observed, books, settings)
        calibrated = calibrate(current, history, settings)
        signals = make_signals(calibrated)
        signal_frames.append(signals)
        diagnostics.append(latency_diagnostics(signals, frame, books, settings))
        status = {}
        for horizon in HORIZONS:
            execution_settings = replace(settings, holding_seconds=horizon)
            evaluation = signals.merge(future_outcomes(frame, execution_settings), on="timestamp", how="left", validate="many_to_one")
            evaluation["horizon"] = horizon
            label_frames.append(evaluation)
            for rule in RULES:
                source = signals[signals.factor.eq(rule)].copy()
                source["horizon"] = horizon
                ledger = simulate_single_day(source, books, execution_settings)
                ledgers.append(ledger)
                status[rule + "/" + str(horizon)] = ledger.status.value_counts().to_dict()
        history.append(current)
        manifest.extend([{"file": "features/" + path.name, "sha256": file_hash(path)},
                         {"file": "normalized/" + path.stem + "/events.parquet", "sha256": file_hash(book_path)}])
        print(json.dumps({"date": path.stem, "signals": {rule: int((signals.factor.eq(rule) & signals.scheduled & signals.signal_status.eq("signal")).sum()) for rule in RULES},
                          "closed": sum(states.get("closed", 0) for states in status.values())}), flush=True)
    signals = pd.concat(signal_frames, ignore_index=True)
    panel = pd.concat(label_frames, ignore_index=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    latency = pd.concat(diagnostics, ignore_index=True)
    evaluation_dates = [path.stem for path in paths[settings.calibration_days:]]
    execution, direction, execution_daily, direction_daily = summarize(panel, ledger, evaluation_dates)
    comparison, sensitivity = paired_comparisons(execution_daily)
    concentration = profit_sensitivity(execution_daily)
    attribution = latency_attribution(ledger, latency)
    for name, data in (("signals", signals), ("observations", pd.concat(history, ignore_index=True)), ("direction_observations", panel)):
        data.to_parquet(output / (name + ".parquet"), index=False)
    for name, data in (("opportunities", ledger), ("execution_comparison", execution), ("direction_comparison", direction),
                       ("execution_daily", execution_daily), ("direction_daily", direction_daily), ("paired_comparisons", comparison),
                       ("leave_one_day_out", sensitivity), ("latency_diagnostics", latency),
                       ("profit_leave_one_day_out", concentration), ("latency_attribution", attribution)):
        data.to_csv(output / (name + ".csv"), index=False)
    counts = ledger.groupby(["date", "factor", "horizon", "status"]).size().rename("count").reset_index()
    counts.to_csv(output / "status_counts.csv", index=False)
    routine = ["closed", "warmup", "no_signal", "unconfirmed"]
    audit = ~ledger.status.isin(routine) | ledger.get("exit_delay_seconds", pd.Series(0, index=ledger.index)).gt(0)
    ledger[audit].to_csv(output / "execution_audit.csv", index=False)
    metadata = {"study": "pressure_events_response_gap_and_confirmation_v1", "settings": asdict(settings),
                "horizons": HORIZONS, "rules": RULES, "comparisons": COMPARISONS, "strict_response_quote_age": STRICT_AGE,
                "dates": [path.stem for path in paths], "evaluation_dates": evaluation_dates,
                "validation": "retrospective_previously_explored_dates", "source_manifest": manifest,
                "protocol_sha256": file_hash(protocol),
                "code_manifest": [{"file": str(path.relative_to(package.parent)), "sha256": file_hash(path)} for path in sorted(package.rglob("*.py"))],
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": version("pyarrow")}}
    (output / "study.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output, execution, direction, comparison, latency, attribution, concentration, metadata)
    return execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    root = args.output_root or resolve_path(load_config(), "results")
    output = root.expanduser() / ("event_response_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    result = run_study(args.data_root.expanduser(), output, StudySettings())
    print(result[["factor", "horizon", "policy_signals", "closed", "mean_net_pnl", "unclosed"]].to_string(index=False))
    print("Report: " + output.name)


if __name__ == "__main__":
    main()
