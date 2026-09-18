from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def volatility_features(price, windows: Sequence[int]) -> Dict[str, np.ndarray]:
    series = pd.Series(price, dtype=float)
    returns = np.log(series / series.shift(1))
    output: Dict[str, np.ndarray] = {}
    for window in windows:
        output["vol_{}s".format(window)] = np.sqrt(returns.pow(2).rolling(window, min_periods=window).sum()).to_numpy()
    base = 30 if 30 in windows else windows[0]
    high = series.rolling(base, min_periods=base).max()
    low = series.rolling(base, min_periods=base).min()
    output["range_volatility"] = np.log(high / low).to_numpy()
    output["return_std"] = returns.rolling(base, min_periods=base).std().to_numpy()
    return output
