from __future__ import annotations

import pandas as pd


def ml_probability(predictions: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    result = predictions[["timestamp"]].copy()
    up = predictions["prob_up"] if "prob_up" in predictions else pd.Series(0, index=predictions.index)
    down = predictions["prob_down"] if "prob_down" in predictions else pd.Series(0, index=predictions.index)
    result["side"] = ""
    result.loc[(up > threshold) & (up >= down), "side"] = "LONG"
    result.loc[(down > threshold) & (down > up), "side"] = "SHORT"
    return result[result["side"] != ""]

