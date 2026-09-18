from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from research_engine.config import config_hash, portable_path
from research_engine.features.options import chain_flow_features, mapped_option_features
from research_engine.features.orderbook import depth_imbalance, orderbook_features
from research_engine.features.price import price_features
from research_engine.features.snapshot import causal_asof, datetime_ns, session_grid
from research_engine.features.trade_flow import trade_flow_features
from research_engine.features.volatility import volatility_features


CORE_COLUMNS = [
    "kind", "symbol", "available_at", "event_ts", "received_at", "price", "volume", "direction",
    "best_bid", "best_bid_size", "best_ask", "best_ask_size", "option_right", "option_strike",
    "option_expiry",
]


def _read(path: Path, filters, *, include_levels: bool = False) -> pd.DataFrame:
    columns = CORE_COLUMNS + (["bid_sizes", "ask_sizes"] if include_levels else [])
    return pq.read_table(str(path), columns=columns, filters=filters).to_pandas()


def _interval_seconds(text: str) -> int:
    if not text.endswith("s"):
        raise ValueError("MVP snapshot_interval must be expressed in seconds, e.g. 1s or 0.5s")
    value = float(text[:-1])
    if value < 1 or int(value) != value:
        raise ValueError("MVP supports whole-second intervals; schema is ready for sub-second extension")
    return int(value)


def _time_features(grid: pd.DatetimeIndex, config: Dict) -> Dict[str, np.ndarray]:
    local = grid.tz_convert(config.get("timezone", "America/New_York"))
    open_hours, open_minutes, open_seconds = map(int, config["session"]["open"].split(":"))
    close_hours, close_minutes, close_seconds = map(int, config["session"]["close"].split(":"))
    open_s = open_hours * 3600 + open_minutes * 60 + open_seconds
    close_s = close_hours * 3600 + close_minutes * 60 + close_seconds
    seconds = ((local.hour * 3600 + local.minute * 60 + local.second) - open_s).to_numpy()
    to_close = close_s - open_s - seconds
    minute = seconds / 60.0
    bucket = np.select(
        [minute < 30, minute < 120, minute < 270, minute < 330],
        ["OPEN", "MORNING", "MIDDAY", "AFTERNOON"], default="POWER_HOUR",
    )
    return {
        "seconds_from_open": seconds.astype(float), "seconds_to_close": to_close.astype(float),
        "minute_from_open": minute, "minute_to_close": to_close / 60.0, "session_bucket": bucket,
    }


def build_feature_frame(normalized_path: Union[str, Path], day: date, config: Dict) -> pd.DataFrame:
    path = Path(normalized_path)
    interval = _interval_seconds(config.get("snapshot_interval", "1s"))
    grid = session_grid(day, config, interval)
    underlying = config.get("underlying", "QQQ.US")
    quote = _read(path, [("kind", "=", "quote"), ("symbol", "=", underlying)])
    depth = _read(path, [("kind", "=", "depth"), ("symbol", "=", underlying)], include_levels=True)
    depth = depth.loc[depth["best_bid"].isna() | depth["best_ask"].isna() |
                      (depth["best_bid"] <= depth["best_ask"])]
    trades = _read(path, [("kind", "=", "trade"), ("symbol", "=", underlying)])
    expiry = day
    option_quotes = _read(path, [("kind", "=", "quote"), ("option_expiry", "=", expiry)])
    option_depths = _read(path, [("kind", "=", "depth"), ("option_expiry", "=", expiry)])
    option_depths = option_depths.loc[option_depths["best_bid"].isna() | option_depths["best_ask"].isna() |
                                      (option_depths["best_bid"] <= option_depths["best_ask"])]
    option_trades = _read(path, [("kind", "=", "trade"), ("option_expiry", "=", expiry)])

    feature_values = {"timestamp": grid}
    if quote.empty:
        feature_values["qqq_last"] = np.full(len(grid), np.nan)
    else:
        feature_values["qqq_last"] = causal_asof(grid, quote["available_at"], {"last": quote["price"]})["last"]
    book_levels = {}
    for level in config["features"]["orderbook"].get("levels", [1]):
        level = int(level)
        if level <= 1:
            continue
        if not depth.empty:
            bid_sizes = depth["bid_sizes"].map(lambda items: float(sum(items[:level])) if items is not None and len(items) else np.nan)
            ask_sizes = depth["ask_sizes"].map(lambda items: float(sum(items[:level])) if items is not None and len(items) else np.nan)
            book_levels[level] = (bid_sizes, ask_sizes)
    if depth.empty:
        book = {name: np.full(len(grid), np.nan) for name in ("bid", "ask", "bid_size", "ask_size", "depth_age_seconds")}
    else:
        book = causal_asof(grid, depth["available_at"], {
            "bid": depth["best_bid"], "ask": depth["best_ask"],
            "bid_size": depth["best_bid_size"], "ask_size": depth["best_ask_size"],
            "book_time_ns": datetime_ns(depth["available_at"]),
        })
        book["depth_age_seconds"] = (grid.astype("int64").to_numpy() - book.pop("book_time_ns")) / 1e9
    feature_values.update(orderbook_features(
        book["bid"], book["ask"], book["bid_size"], book["ask_size"],
        config["features"]["orderbook"]["windows"],
    ))
    for level, (bid_values, ask_values) in book_levels.items():
        aligned = causal_asof(grid, depth["available_at"], {"bid_size": bid_values, "ask_size": ask_values})
        feature_values["bid_size_l{}".format(level)] = aligned["bid_size"]
        feature_values["ask_size_l{}".format(level)] = aligned["ask_size"]
        feature_values["depth_imbalance_l{}".format(level)] = depth_imbalance(aligned["bid_size"], aligned["ask_size"])
    feature_values["qqq_depth_age_seconds"] = book["depth_age_seconds"]

    reference = np.where(np.isfinite(feature_values["qqq_mid"]), feature_values["qqq_mid"], feature_values["qqq_last"])
    feature_values.update(price_features(
        reference, config["features"]["returns"]["windows"],
        config["features"]["momentum"]["windows"], config["features"]["ranges"]["windows"],
    ))
    feature_values.update(volatility_features(reference, config["features"]["volatility"]["windows"]))
    if not trades.empty:
        flow = trade_flow_features(
            grid, trades["available_at"], trades["volume"], trades["direction"],
            config["features"]["trade_flow"]["windows"],
        )
        feature_values.update(flow)

    option_config = config["options"]
    mapped = mapped_option_features(
        grid, reference, option_quotes, option_depths, option_trades,
        option_config["relative_strikes"], option_config["return_windows"],
        option_config["volume_windows"], int(option_config["flow_window"]),
        int(option_config["volume_burst_lookback"]),
    )
    mapped.update(chain_flow_features(
        grid, reference, option_trades, int(option_config["flow_window"]),
        float(option_config["near_atm_distance"]),
    ))
    mapped.update(_time_features(grid, config))
    # Build once to avoid pandas' column-fragmentation penalty for hundreds of features.
    feature_values.update(mapped)
    return pd.DataFrame(feature_values)


def write_feature_outputs(
    normalized_path: Union[str, Path], day: date, snapshot_output: Union[str, Path],
    feature_output: Union[str, Path], config: Dict, *, force: bool = False,
) -> Dict:
    normalized_path, snapshot_output, feature_output = map(Path, (normalized_path, snapshot_output, feature_output))
    normalized_meta_path = normalized_path.with_suffix(normalized_path.suffix + ".meta.json")
    normalized_meta = json.loads(normalized_meta_path.read_text())
    expected = {"source_hash": normalized_meta["source_hash"], "config_hash": config_hash(config, "features"),
                "date": day.isoformat(), "feature_version": 7}
    meta_path = feature_output.with_suffix(feature_output.suffix + ".meta.json")
    if not force and feature_output.exists() and snapshot_output.exists() and meta_path.exists():
        current = json.loads(meta_path.read_text())
        if all(current.get(k) == v for k, v in expected.items()):
            return dict(expected, cached=True, feature_output=str(feature_output), snapshot_output=str(snapshot_output))
    snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_feature_frame(normalized_path, day, config)
    snapshot_columns = [
        "timestamp", "qqq_last", "qqq_bid", "qqq_ask", "qqq_mid", "qqq_spread",
        "qqq_spread_bps", "bid_size", "ask_size",
    ]
    frame[snapshot_columns].to_parquet(snapshot_output, index=False, compression="zstd")
    frame.to_parquet(feature_output, index=False, compression="zstd")
    result = dict(expected, cached=False, rows=len(frame), columns=len(frame.columns),
                  feature_output=str(feature_output), snapshot_output=str(snapshot_output))
    meta_path.write_text(json.dumps(dict(result, feature_output=portable_path(feature_output),
                                        snapshot_output=portable_path(snapshot_output)), indent=2) + "\n", encoding="utf-8")
    return result
