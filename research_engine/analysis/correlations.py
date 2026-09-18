from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def correlations(dataset: pd.DataFrame, features: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    day = pd.to_datetime(dataset["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    for feature in features:
        if feature not in dataset:
            continue
        for label in labels:
            if label not in dataset:
                continue
            for scope, frame in [("overall", dataset)] + [
                (str(d), dataset.loc[day == d]) for d in sorted(day.unique())
            ]:
                values = frame[[feature, label]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) < 3 or values[feature].nunique() < 2 or values[label].nunique() < 2:
                    continue
                rows.append({
                    "feature": feature, "label": label, "scope": scope,
                    "sample_count": len(values),
                    "pearson": values[feature].corr(values[label], method="pearson"),
                    "spearman": values[feature].corr(values[label], method="spearman"),
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    daily = result[result["scope"] != "overall"].groupby(["feature", "label"])[["pearson", "spearman"]]
    stability = daily.agg(["mean", "std"])
    stability.columns = ["ic_{}_{}".format(name, stat) for name, stat in stability.columns]
    stability = stability.reset_index()
    return result.merge(stability, on=["feature", "label"], how="left")

