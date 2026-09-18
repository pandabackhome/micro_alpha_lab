from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from research_engine.features.snapshot import bucket_numeric, trailing_change, trailing_sum


def trade_flow_features(
    grid: pd.DatetimeIndex,
    times,
    volumes,
    directions,
    windows: Sequence[int],
    prefix: str = "",
) -> Dict[str, np.ndarray]:
    volumes = np.asarray(volumes, dtype=float)
    directions = np.asarray(directions, dtype=object)
    one = np.ones(len(volumes), dtype=float)
    buy = bucket_numeric(grid, times, np.where(directions == "TradeDirection.Up", volumes, 0.0))
    sell = bucket_numeric(grid, times, np.where(directions == "TradeDirection.Down", volumes, 0.0))
    neutral = bucket_numeric(grid, times, np.where(directions == "TradeDirection.Neutral", volumes, 0.0))
    buy_count = bucket_numeric(grid, times, np.where(directions == "TradeDirection.Up", one, 0.0))
    sell_count = bucket_numeric(grid, times, np.where(directions == "TradeDirection.Down", one, 0.0))
    count = bucket_numeric(grid, times, one)
    maximum = bucket_numeric(grid, times, volumes, reduction="max")
    output: Dict[str, np.ndarray] = {}
    for window in windows:
        suffix = "_{}s".format(window)
        b, s, n = trailing_sum(buy, window), trailing_sum(sell, window), trailing_sum(neutral, window)
        bc, sc, total_count = (
            trailing_sum(buy_count, window),
            trailing_sum(sell_count, window),
            trailing_sum(count, window),
        )
        total = b + s + n
        denominator = b + s
        imbalance = np.divide(b - s, denominator, out=np.zeros_like(denominator), where=denominator != 0)
        max_window = pd.Series(maximum).rolling(window, min_periods=1).max().to_numpy()
        output[prefix + "buy_volume" + suffix] = b
        output[prefix + "sell_volume" + suffix] = s
        output[prefix + "neutral_volume" + suffix] = n
        output[prefix + "buy_count" + suffix] = bc
        output[prefix + "sell_count" + suffix] = sc
        output[prefix + "total_volume" + suffix] = total
        output[prefix + "trade_count" + suffix] = total_count
        output[prefix + "avg_trade_size" + suffix] = np.divide(
            total, total_count, out=np.zeros_like(total), where=total_count != 0
        )
        output[prefix + "max_trade_size" + suffix] = max_window
        output[prefix + "trade_imbalance" + suffix] = imbalance
    primary = 5 if 5 in windows else windows[0]
    imbalance = output[prefix + "trade_imbalance_{}s".format(primary)]
    change = trailing_change(imbalance)
    output[prefix + "trade_imbalance_change"] = change
    output[prefix + "trade_imbalance_acceleration"] = trailing_change(change)
    return output

