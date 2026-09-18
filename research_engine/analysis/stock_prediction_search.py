"""Previous-date underlying return forecasts gated by contemporaneous option costs."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, future_outcomes, simulate_single_day
from research_engine.analysis.event_response import estimated_cost
from research_engine.analysis.stock_strategy_search import COLUMNS, HORIZONS, aggregate, fresh_gate, pick, stock_observations

MODELS = ("ridge", "tree")
MARGINS = (1, 2)
FEATURES = ["r5", "r30", "r60", "r180", "deviation", "range60", "log_flow", "log_volume", "past_vol",
            "session_fraction", "morning", "morning_r60", "morning_r180", "morning_deviation", "volatility_r60"]
SIGNAL_COLUMNS = ["timestamp", "date", "half_hour", "spot", "past_vol", "session_close", "scheduled", "valid300"]


def model_features(current):
    result = current.copy()
    result["log_flow"] = np.sign(result.flow30) * np.log1p(result.flow30.abs())
    result["log_volume"] = np.log1p(result.volume5.where(result.volume5.ge(0)))
    result["session_fraction"] = result.seconds_from_open / 23400
    result["morning"] = result.seconds_from_open.lt(7200).astype(float)
    result["morning_r60"] = result.morning * result.r60
    result["morning_r180"] = result.morning * result.r180
    result["morning_deviation"] = result.morning * result.deviation
    result["volatility_r60"] = result.past_vol * result.r60
    return result


def new_model(name):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.))
    if name == "tree":
        return HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=.05,
                                            min_samples_leaf=100, l2_regularization=10.,
                                            early_stopping=False, random_state=42)
    raise ValueError("unknown model")


def forecast(current, history, horizon, model_name, min_samples=1000):
    prediction = np.full(len(current), np.nan)
    details = dict(model=model_name, horizon=horizon, training_samples=0, history_first_date=None, history_last_date=None)
    if len(history) < 5:
        return prediction, details
    past = pd.concat(history[-5:], ignore_index=True)
    if past.date.max() >= current.date.min() or past.date.nunique() != 5:
        raise ValueError("training requires five distinct earlier dates")
    target = "target_" + str(horizon)
    eligible = past.valid300 & np.isfinite(past[FEATURES]).all(axis=1) & np.isfinite(past[target])
    train = past[eligible]
    details.update(training_samples=len(train), history_first_date=past.date.min(), history_last_date=past.date.max())
    if len(train) < min_samples:
        return prediction, details
    model = new_model(model_name)
    model.fit(train[FEATURES], train[target])
    good = current.valid300 & np.isfinite(current[FEATURES]).all(axis=1)
    if good.any():
        prediction[good.to_numpy()] = model.predict(current.loc[good, FEATURES])
    return prediction, details


def cost_signals(current, prediction, books, settings, model_name, margin, details):
    result = current[SIGNAL_COLUMNS].copy()
    result["forecast_bps"] = prediction
    result["factor"] = model_name + "_cost" + str(margin)
    result["regime"] = "model"
    result["signal_status"] = "no_signal"
    result["direction"] = 0
    result["history_last_date"] = details["history_last_date"]
    result["training_samples"] = details["training_samples"]
    result["cost_estimate"] = np.nan
    result["predicted_option_move"] = np.nan
    result["decision_symbol"] = None
    for idx, row in result.iterrows():
        if not row.valid300:
            result.loc[idx, "signal_status"] = "invalid_spot_window"
            continue
        if not np.isfinite(row.forecast_bps):
            result.loc[idx, "signal_status"] = "forecast_unavailable"
            continue
        if row.forecast_bps == 0:
            continue
        direction = int(np.sign(row.forecast_bps))
        contract = books.select_leg(row.timestamp, row.spot, "CALL" if direction > 0 else "PUT", settings.max_atm_distance)
        if contract is None:
            result.loc[idx, "signal_status"] = "decision_contract_unavailable"
            continue
        quote, reason = books.quote(contract[1], row.timestamp, 1.)
        if quote is None:
            result.loc[idx, "signal_status"] = "decision_quote_unavailable"
            continue
        predicted = .5 * row.spot * abs(row.forecast_bps) / 10000
        cost = estimated_cost(quote, settings)
        result.loc[idx, ["cost_estimate", "predicted_option_move", "decision_symbol"]] = (cost, predicted, contract[1])
        if predicted > margin * cost:
            result.loc[idx, ["direction", "signal_status"]] = (direction, "signal")
    return result


def run(data_root, output):
    settings = StudySettings()
    paths = sorted((data_root / "features").glob("*.parquet"))
    if len(paths) < 14:
        raise ValueError("requires at least fourteen dates for the fixed split")
    dates = [path.stem for path in paths]
    discovery_dates, check_dates = dates[5:10], dates[10:]
    output.mkdir(parents=True, exist_ok=False)
    (output / "opportunities").mkdir()
    history, training, daily, manifests, prediction_rows = [], [], [], [], []
    for path in paths:
        frame = pd.read_parquet(path, columns=COLUMNS)
        current = model_features(stock_observations(frame, settings))
        if set(current.date) != {path.stem}:
            raise ValueError("feature date mismatch")
        manifests.append(dict(file="features/" + path.name, sha256=file_hash(path)))
        if len(history) >= 5:
            source = data_root / "normalized" / path.stem / "events.parquet"
            books = read_books(source, path.stem)
            manifests.append(dict(file="normalized/" + path.stem + "/events.parquet", sha256=file_hash(source)))
            ledgers = []
            for horizon in HORIZONS:
                execution = replace(settings, holding_seconds=horizon)
                for model_name in MODELS:
                    prediction, details = forecast(current, history, horizon, model_name)
                    training.append(dict(date=path.stem, **details))
                    for margin in MARGINS:
                        signals = cost_signals(current, prediction, books, execution, model_name, margin, details)
                        signals["horizon"] = horizon
                        prediction_rows.append(signals)
                        ledger = simulate_single_day(signals, books, execution, entry_gate=fresh_gate(books))
                        ledgers.append(ledger)
                        metrics = execution_metrics(ledger)
                        metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
                        daily.append(dict(date=path.stem, factor=model_name + "_cost" + str(margin), horizon=horizon, **metrics))
            day = pd.concat(ledgers, ignore_index=True)
            day.to_parquet(output / "opportunities" / (path.stem + ".parquet"), index=False)
            print(json.dumps(dict(date=path.stem, closed=int(day.status.eq("closed").sum()))), flush=True)
        # Future outcomes are appended only to the completed day's training cache.
        # Today's prediction above consumes history, never these outcome columns.
        completed = current.copy()
        for horizon in HORIZONS:
            labels = future_outcomes(frame, replace(settings, holding_seconds=horizon))[["timestamp", "delayed_return_bps"]]
            completed = completed.merge(labels.rename(columns={"delayed_return_bps": "target_" + str(horizon)}), on="timestamp", validate="one_to_one")
        history.append(completed)
    daily = pd.DataFrame(daily)
    discovery = aggregate(daily, discovery_dates)
    selected = pick(discovery)
    checking = aggregate(daily, check_dates).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    overall = aggregate(daily, dates[5:])
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", checking),
                          ("all_dates", overall), ("training_audit", pd.DataFrame(training))]:
        values.to_csv(output / (name + ".csv"), index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(output / "signals.parquet", index=False)
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/stock_prediction_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    metadata = dict(study="past_date_stock_forecast_cost_gate", models=MODELS, margins=MARGINS, horizons=HORIZONS,
                    features=FEATURES, settings=asdict(settings), dates=dates, discovery_dates=discovery_dates,
                    check_dates=check_dates, trials=12, source_manifest=manifests, protocol_sha256=file_hash(protocol),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# 正股幅度模型开仓期权", "", "固定模型，过去5天训练，当天冻结。下面均包含价差、手续费和滑点；压力损益包含未平仓零回收假设。",
             "", "## 前段全部结果", "", "```", discovery.to_string(index=False), "```", "",
             "## 前段选出候选的后段结果", "", "```", checking.to_string(index=False), "```", "",
             "这些日期此前已看过，多轮探索后本轮仍不是独立验证。期权响应系数0.5仅为成本门槛代理，实际成交另按同合约报价回测。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Discovery:"); print(discovery.to_string(index=False))
    print("Selected check:"); print(checking.to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("stock_prediction_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, output)


if __name__ == "__main__":
    main()
