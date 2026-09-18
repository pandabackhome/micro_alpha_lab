from __future__ import annotations

from datetime import date, datetime, time
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pytz


NAT_INT = np.iinfo(np.int64).min


def session_grid(day: date, config: Dict, interval_seconds: int = 1) -> pd.DatetimeIndex:
    zone = pytz.timezone(config.get("timezone", "America/New_York"))
    open_t = time.fromisoformat(config["session"]["open"])
    close_t = time.fromisoformat(config["session"]["close"])
    start = zone.localize(datetime.combine(day, open_t)).astimezone(pytz.UTC)
    end = zone.localize(datetime.combine(day, close_t)).astimezone(pytz.UTC)
    return pd.date_range(start, end, freq="{}s".format(interval_seconds), tz="UTC")


def datetime_ns(values) -> np.ndarray:
    # Arrow timestamps arrive with microsecond precision, while the grid is
    # nanosecond based. Comparing their raw integer representations leaks the
    # final observation into the first grid row; normalize units explicitly.
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns").asi8


def causal_asof(
    grid: pd.DatetimeIndex,
    available_at,
    fields: Mapping[str, Sequence],
) -> Dict[str, np.ndarray]:
    """Backward as-of: a source row can affect only grid timestamps >= availability."""
    grid_ns = grid.astype("int64").to_numpy()
    source_ns = datetime_ns(available_at)
    order = np.argsort(source_ns, kind="stable")
    source_ns = source_ns[order]
    indices = np.searchsorted(source_ns, grid_ns, side="right") - 1
    valid = indices >= 0
    result = {}
    for name, values in fields.items():
        source = np.asarray(values)[order]
        if np.issubdtype(source.dtype, np.number):
            target = np.full(len(grid), np.nan, dtype=float)
        else:
            target = np.full(len(grid), None, dtype=object)
        target[valid] = source[indices[valid]]
        result[name] = target
    return result


def bucket_numeric(grid: pd.DatetimeIndex, times, values, reduction: str = "sum") -> np.ndarray:
    grid_ns = grid.astype("int64").to_numpy()
    event_ns = datetime_ns(times)
    index = np.searchsorted(grid_ns, event_ns, side="left")
    valid = (index >= 0) & (index < len(grid)) & (event_ns <= grid_ns[-1])
    result = np.zeros(len(grid), dtype=float)
    if reduction == "sum":
        np.add.at(result, index[valid], np.asarray(values, dtype=float)[valid])
    elif reduction == "max":
        np.maximum.at(result, index[valid], np.asarray(values, dtype=float)[valid])
    else:
        raise ValueError("unsupported reduction: {}".format(reduction))
    return result


def trailing_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(np.nan_to_num(values, nan=0.0))))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return cumulative[end] - cumulative[start]


def trailing_change(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    result[1:] = values[1:] - values[:-1]
    return result
