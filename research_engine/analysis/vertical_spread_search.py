"""Protected option credit spreads driven only by underlying stock events."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.event_reversal_search import crossing_signals
from research_engine.analysis.frozen_dataset import research_inputs, DEFAULT_MANIFEST
from research_engine.analysis.stock_strategy_search import COLUMNS, aggregate, calibrate, pick, stock_observations


def select_spread(books, when, spot, direction, settings):
    right = "PUT" if direction > 0 else "CALL"
    short = books.select_leg(when, spot, right, settings.max_atm_distance)
    if short is None:
        return None
    long_strike = short[0] + (-1. if direction > 0 else 1.)
    long = [item for item in books.contracts if item[1] == right and item[3] <= when.value and abs(item[0] - long_strike) < 1e-9]
    if not long:
        return None
    return dict(right=right, short_symbol=short[1], long_symbol=long[0][2],
                short_strike=short[0], long_strike=long_strike, width=1.)


def spread_quotes(books, contract, when, signal_time=None):
    short, sr = books.quote(contract["short_symbol"], when, 1.)
    long, lr = books.quote(contract["long_symbol"], when, 1.)
    if short is None or long is None:
        return None, "short:" + sr + ";long:" + lr
    if signal_time is not None and any(when - pd.Timedelta(seconds=q["age"]) <= signal_time for q in (short, long)):
        return None, "leg_quote_not_updated_after_signal"
    return (short, long), "ok"


def simulate_spreads(signals, books, settings):
    rows, occupied = [], None
    slip, fee, multiplier = settings.slippage_bps / 10000, settings.commission_per_contract, settings.multiplier
    for signal in signals[signals.scheduled].sort_values("timestamp").to_dict("records"):
        row = dict(signal, status=signal["signal_status"], reason="", policy_eligible=False, horizon=settings.holding_seconds)
        rows.append(row)
        if signal["signal_status"] != "signal":
            continue
        row["policy_eligible"] = True
        when, close = signal["timestamp"], signal["session_close"]
        if occupied is not None and when < occupied:
            row.update(status="busy", reason="previous spread remains open")
            continue
        entry_time = when + pd.Timedelta(seconds=settings.latency_seconds)
        deadline = entry_time + pd.Timedelta(seconds=settings.holding_seconds)
        if deadline > close:
            row.update(status="session_end", reason="insufficient holding time")
            continue
        contract = select_spread(books, when, signal["spot"], signal["direction"], settings)
        if contract is None:
            row.update(status="missing_contract", reason="missing observed short or protection leg")
            continue
        row.update(contract, entry_time=entry_time, exit_due=deadline)
        quotes, reason = spread_quotes(books, contract, entry_time, signal_time=when)
        if quotes is None:
            row.update(status="entry_rejected", reason=reason)
            continue
        short, long = quotes
        short_fill, long_fill = short["bid"] * (1 - slip), long["ask"] * (1 + slip)
        credit = short_fill - long_fill
        cash = credit * multiplier - 2 * fee
        if cash <= 0 or credit > contract["width"]:
            row.update(status="entry_rejected", reason="credit outside permitted price limits")
            continue
        risk = contract["width"] * multiplier - cash + 2 * fee
        row.update(entry_short_bid=short["bid"], entry_long_ask=long["ask"], entry_short_fill=short_fill,
                   entry_long_fill=long_fill, entry_credit=cash, maximum_loss_model=risk,
                   entry_short_age=short["age"], entry_long_age=long["age"])
        limit = min(close, deadline + pd.Timedelta(seconds=settings.max_exit_delay_seconds))
        exit_quotes = None
        for exit_time in pd.date_range(deadline, limit, freq="1s"):
            exit_quotes, reason = spread_quotes(books, contract, exit_time)
            if exit_quotes is not None:
                if exit_quotes[0]["ask"] < exit_quotes[1]["bid"]:
                    exit_quotes, reason = None, "inconsistent_vertical_exit_quote"
                    continue
                break
        if exit_quotes is None:
            row.update(status="unclosed", reason=reason, last_exit_check=limit)
            occupied = close
            continue
        short, long = exit_quotes
        short_fill, long_fill = short["ask"] * (1 + slip), long["bid"] * (1 - slip)
        debit = (short_fill - long_fill) * multiplier + 2 * fee
        net = cash - debit
        row.update(status="closed", exit_time=exit_time, exit_short_ask=short["ask"], exit_long_bid=long["bid"],
                   exit_short_fill=short_fill, exit_long_fill=long_fill, exit_debit=debit,
                   exit_short_age=short["age"], exit_long_age=long["age"], net_pnl=net,
                   net_on_model_risk=net / risk, commission=4 * fee,
                   holding_seconds=(exit_time-entry_time).total_seconds(),
                   exit_delay_seconds=(exit_time-deadline).total_seconds())
        occupied = exit_time
    return pd.DataFrame(rows)


def metrics(ledger):
    closed = ledger[ledger.status.eq("closed")]
    unresolved = ledger[ledger.status.eq("unclosed")]
    net = float(closed.net_pnl.sum()) if len(closed) else 0.
    risk = float(unresolved.maximum_loss_model.sum()) if len(unresolved) else 0.
    return dict(policy_signals=int(ledger.policy_eligible.sum()), closed=len(closed), unclosed=len(unresolved),
                realized_net_pnl=net, mean_net_pnl=net/len(closed) if len(closed) else None,
                unclosed_debit=risk, net_with_unclosed_zero_recovery=net-risk,
                entry_rejected=int(ledger.status.eq("entry_rejected").sum()))


def run(archive_data, local_data, output):
    settings = StudySettings()
    paths, dataset = research_inputs(archive_data, local_data)
    dates = sorted(paths)
    if len(dates) < 16:
        raise ValueError("requires the original fourteen and two added dates")
    discovery_dates, check_dates, added_dates = dates[5:10], dates[10:14], dates[14:]
    output.mkdir(parents=True, exist_ok=False)
    history, manifests, ledgers, daily = [], [], [], []
    for day in dates:
        origin, path, source = paths[day]
        frame = pd.read_parquet(path, columns=COLUMNS)
        ordinary = stock_observations(frame, settings)
        manifests.append(dict(origin=origin, file="features/" + path.name, sha256=file_hash(path)))
        if len(history) >= 5:
            dense = calibrate(stock_observations(frame, settings, sampling_seconds=1), history, settings)
            cross = crossing_signals(dense)
            recovery = crossing_signals(dense, mode="recovery")
            books = read_books(source, day)
            manifests.append(dict(origin=origin, file="normalized/" + day + "/events.parquet", sha256=file_hash(source)))
            for factor, signals, flip in [("cross_reversal", cross, False), ("cross_momentum", cross, True), ("recovery_reversal", recovery, False)]:
                prepared = signals.assign(factor=factor)
                if flip:
                    prepared["direction"] *= -1
                for horizon in (180, 300):
                    ledger = simulate_spreads(prepared, books, replace(settings, holding_seconds=horizon))
                    ledgers.append(ledger)
                    daily.append(dict(date=day, factor=factor, horizon=horizon, **metrics(ledger)))
            print("Replayed protected spreads " + day, flush=True)
        history.append(ordinary)
    daily = pd.DataFrame(daily)
    discovery = aggregate(daily, discovery_dates)
    selected = pick(discovery)
    check = aggregate(daily, check_dates).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    added = aggregate(daily, added_dates).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", check),
                          ("added_dates_check", added), ("all_dates", aggregate(daily, dates[5:]))]:
        values.to_csv(output / (name + ".csv"), index=False)
    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_parquet(output / "opportunities.parquet", index=False)
    ledger.groupby(["date", "factor", "horizon", "status"]).size().rename("count").to_csv(output / "status_counts.csv")
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/vertical_spread_protocol.md"
    (output / "protocol.md").write_bytes(protocol.read_bytes())
    metadata = dict(study="stock_events_protected_credit_spreads", settings=asdict(settings), trials=6,
                    dataset_manifest_sha256=file_hash(DEFAULT_MANIFEST),
                    dates=dates, discovery_dates=discovery_dates, check_dates=check_dates, added_dates=added_dates,
                    source_manifest=manifests, protocol_sha256=file_hash(protocol),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# 正股事件与有保护腿信用价差", "", "两腿开平共收四次手续费。未退出时扣模型最大损失计算压力损益。",
        "为复用汇总格式，unclosed_debit 表示未退出价差的模型风险额，并非实际入场付出的现金。",
        "双腿同期报价不保证真实组合单成交；未模拟分腿风险、队列及提前指派。", "",
        "## 前段全部组合", "", "```", discovery.to_string(index=False), "```", "",
        "## 选出方案的旧后段", "", "```", check.to_string(index=False), "```", "",
        "## 选出方案的新增日期", "", "```", added.to_string(index=False), "```", "",
        "新增日期此前已用于其他规则，本轮不再把它们称作完全未查看的数据。", ""]), encoding="utf-8")
    print(discovery.to_string(index=False))
    print("Old check:"); print(check.to_string(index=False))
    print("Added dates:"); print(added.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data", type=Path, required=True)
    parser.add_argument("--local-data", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    output = Path("results/research") / ("vertical_spread_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data, args.local_data, output)


if __name__ == "__main__":
    main()
