"""Predict executable option net PnL using only causal underlying features."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.stock_prediction_search import FEATURES, MODELS, SIGNAL_COLUMNS, forecast, model_features
from research_engine.analysis.stock_strategy_search import COLUMNS, HORIZONS, aggregate, fresh_gate, pick, stock_observations

BUFFERS = (0, 3)
RIGHTS = {"CALL": 1, "PUT": -1}


def decision_quotes(current, books, settings):
    """Only contemporaneously observable availability enters a decision."""
    result = current[["timestamp"]].copy()
    for right in RIGHTS:
        symbols = []
        for row in current.itertuples():
            symbol = None
            if row.valid300 and np.isfinite(row.spot):
                contract = books.select_leg(row.timestamp, row.spot, right, settings.max_atm_distance)
                if contract is not None:
                    quote, _ = books.quote(contract[1], row.timestamp, 1.)
                    if quote is not None and quote["mid"] >= settings.min_entry_mid:
                        symbol = contract[1]
            symbols.append(symbol)
        result[right + "_symbol"] = symbols
    return result


def episode_labels(current, quotes, books, settings):
    """Independent training episodes; never a portfolio equity calculation."""
    completed = current.copy()
    rows = []
    gate = fresh_gate(books)
    for horizon in HORIZONS:
        execution = replace(settings, holding_seconds=horizon)
        for right, direction in RIGHTS.items():
            target = "target_" + right + "_" + str(horizon)
            completed[target] = np.nan
            for idx, signal in current[SIGNAL_COLUMNS].iterrows():
                if not signal.valid300 or pd.isna(quotes.loc[idx, right + "_symbol"]):
                    rows.append(dict(timestamp=signal.timestamp, date=signal.date, right=right,
                                     horizon=horizon, status="decision_unavailable", target_net=np.nan))
                    continue
                one = pd.DataFrame([dict(signal, factor="training_episode", regime="training",
                                         signal_status="signal", direction=direction)])
                episode = simulate_single_day(one, books, execution, entry_gate=gate).iloc[0].to_dict()
                value = np.nan
                if episode["status"] == "closed":
                    value = episode["net_pnl"]
                elif episode["status"] == "unclosed":
                    value = -episode["entry_debit"]
                completed.loc[idx, target] = value
                episode.update(right=right, horizon=horizon, target_net=value)
                rows.append(episode)
    return completed, pd.DataFrame(rows)


def forecasts(current, history, horizon, model_name, min_samples=500):
    predictions, audits = {}, []
    for right in RIGHTS:
        target = "target_" + right + "_" + str(horizon)
        past = [day.rename(columns={target: "target_" + str(horizon)}) for day in history[-5:]]
        prediction, details = forecast(current, past, horizon, model_name, min_samples=min_samples)
        details["right"] = right
        predictions[right] = prediction
        audits.append(details)
    return predictions, audits


def choose_signals(current, quotes, predictions, model_name, buffer):
    result = current[SIGNAL_COLUMNS].copy()
    result["factor"] = model_name + "_net" + str(buffer)
    result["regime"] = "option_net_model"
    result["direction"] = 0
    result["signal_status"] = "no_signal"
    result["forecast_net"] = np.nan
    result["decision_symbol"] = None
    for right in RIGHTS:
        result["forecast_" + right] = predictions[right]
    for idx, row in result.iterrows():
        if not row.valid300:
            result.loc[idx, "signal_status"] = "invalid_spot_window"
            continue
        available = [(row["forecast_" + right], right) for right in RIGHTS
                     if np.isfinite(row["forecast_" + right]) and pd.notna(quotes.loc[idx, right + "_symbol"])]
        if not available:
            result.loc[idx, "signal_status"] = "forecast_or_quote_unavailable"
            continue
        # An exact tie always chooses CALL; no future execution information is read.
        value, right = max(available, key=lambda item: item[0])
        result.loc[idx, "forecast_net"] = value
        if value > buffer:
            result.loc[idx, ["direction", "signal_status", "decision_symbol"]] = (
                RIGHTS[right], "signal", quotes.loc[idx, right + "_symbol"])
    return result


def run(data_root, output):
    settings = StudySettings()
    paths = sorted((data_root / "features").glob("*.parquet"))
    if len(paths) < 14:
        raise ValueError("requires at least fourteen dates")
    dates = [path.stem for path in paths]
    discovery_dates, check_dates = dates[5:10], dates[10:]
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("opportunities", "training_episodes", "training_features"):
        (output / folder).mkdir()
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/option_target_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    history, daily, audits, manifests, signals_all, label_counts = [], [], [], [], [], []
    for path in paths:
        frame = pd.read_parquet(path, columns=COLUMNS)
        current = model_features(stock_observations(frame, settings))
        current = current[current.scheduled].reset_index(drop=True)
        if set(current.date) != {path.stem}:
            raise ValueError("feature date mismatch")
        source = data_root / "normalized" / path.stem / "events.parquet"
        books = read_books(source, path.stem)
        manifests += [dict(file="features/" + path.name, sha256=file_hash(path)),
                      dict(file="normalized/" + path.stem + "/events.parquet", sha256=file_hash(source))]
        quotes = decision_quotes(current, books, settings)
        if len(history) >= 5:
            ledgers = []
            for horizon in HORIZONS:
                execution = replace(settings, holding_seconds=horizon)
                for model_name in MODELS:
                    predictions, details = forecasts(current, history, horizon, model_name)
                    audits.extend(dict(date=path.stem, **detail) for detail in details)
                    for buffer in BUFFERS:
                        signals = choose_signals(current, quotes, predictions, model_name, buffer)
                        signals["horizon"] = horizon
                        signals["history_last_date"] = history[-1].date.iloc[0]
                        signals_all.append(signals)
                        ledger = simulate_single_day(signals, books, execution, entry_gate=fresh_gate(books))
                        metrics = execution_metrics(ledger)
                        metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
                        daily.append(dict(date=path.stem, factor=model_name + "_net" + str(buffer), horizon=horizon, **metrics))
                        ledgers.append(ledger)
            pd.concat(ledgers, ignore_index=True).to_parquet(output / "opportunities" / (path.stem + ".parquet"), index=False)
        # This day's targets are constructed only after its predictions and replay.
        completed, episodes = episode_labels(current, quotes, books, settings)
        episodes.to_parquet(output / "training_episodes" / (path.stem + ".parquet"), index=False)
        completed.to_parquet(output / "training_features" / (path.stem + ".parquet"), index=False)
        label_counts.append(episodes.groupby(["date", "right", "horizon", "status"]).size().rename("count").reset_index())
        history.append(completed)
        print(json.dumps(dict(date=path.stem, training_episodes=len(episodes), closed=int(episodes.status.eq("closed").sum()))), flush=True)
    daily = pd.DataFrame(daily)
    discovery = aggregate(daily, discovery_dates)
    selected = pick(discovery)
    checking = aggregate(daily, check_dates).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", checking),
                          ("all_dates", aggregate(daily, dates[5:])), ("training_audit", pd.DataFrame(audits)),
                          ("training_status_counts", pd.concat(label_counts, ignore_index=True))]:
        values.to_csv(output / (name + ".csv"), index=False)
    pd.concat(signals_all, ignore_index=True).to_parquet(output / "signals.parquet", index=False)
    metadata = dict(study="stock_features_option_net_targets", models=MODELS, buffers=BUFFERS, horizons=HORIZONS,
                    features=FEATURES, settings=asdict(settings), dates=dates, discovery_dates=discovery_dates,
                    check_dates=check_dates, trials=12, source_manifest=manifests, protocol_sha256=file_hash(protocol),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# 正股特征直接预测期权净收益", "", "过去5日训练，当天冻结，扣除价差、手续费和滑点。未平仓另以零回收计算压力损益。",
             "训练情景可重叠，组合最多同时一张；不能把训练情景收益相加作为组合收益。", "",
             "## 前段全部结果", "", "```", discovery.to_string(index=False), "```", "",
             "## 前段选出的候选在后段的结果", "", "```", checking.to_string(index=False), "```", "",
             "全部日期已用于探索，不是独立验证。不可入场情景缺少训练目标，报价可得性仍可能影响模型。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(discovery.to_string(index=False))
    print("Selected check:"); print(checking.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("option_target_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, output)


if __name__ == "__main__":
    main()
