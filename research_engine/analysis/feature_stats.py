from __future__ import annotations

import pandas as pd


def feature_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    result = numeric.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).T
    result["missing_pct"] = numeric.isna().mean() * 100.0
    return result.reset_index().rename(columns={"index": "feature"})

