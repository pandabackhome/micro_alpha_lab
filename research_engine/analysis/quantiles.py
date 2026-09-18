from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def quantile_analysis(
    dataset: pd.DataFrame,
    feature: str,
    labels: Sequence[str] = ("future_ret_10s", "future_ret_30s", "future_ret_60s"),
    bins: int = 10,
) -> pd.DataFrame:
    if feature not in dataset:
        raise KeyError("feature not found: {}".format(feature))
    finite = dataset[feature].replace([np.inf, -np.inf], np.nan)
    valid = finite.notna()
    groups = pd.Series(index=dataset.index, dtype=float)
    if valid.sum() and finite[valid].nunique() > 1:
        # Preserve equal values as one group. Ranking ties by row order can
        # invent a false quantile effect for zero-inflated flow features.
        groups.loc[valid] = pd.qcut(
            finite[valid], q=min(bins, int(valid.sum())), labels=False, duplicates="drop"
        ).astype(float)
    work = dataset.copy()
    work["quantile"] = groups
    columns = [x for x in labels if x in work]
    excursion = [x for x in ("mfe_30s", "mae_30s", "mfe_60s", "mae_60s") if x in work]
    rows = []
    for key, segment in work.dropna(subset=["quantile"]).groupby("quantile"):
        row = {
            "feature": feature,
            "quantile": int(key) + 1,
            "sample_count": int(len(segment)),
            "feature_min": float(segment[feature].min()),
            "feature_max": float(segment[feature].max()),
            "feature_median": float(segment[feature].median()),
        }
        for label in columns:
            row[label + "_mean"] = segment[label].mean()
            row[label + "_median"] = segment[label].median()
            row[label + "_positive_rate"] = (segment[label].dropna() > 0).mean()
        for label in excursion:
            row[label + "_mean"] = segment[label].mean()
        rows.append(row)
    return pd.DataFrame(rows)
