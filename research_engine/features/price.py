from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def price_features(price, return_windows: Sequence[int], momentum_windows: Sequence[int], range_windows: Sequence[int]):
    series = pd.Series(price, dtype=float)
    output: Dict[str, np.ndarray] = {}
    for window in return_windows:
        output["ret_{}s".format(window)] = np.log(series / series.shift(window)).to_numpy()
    for window in momentum_windows:
        output["momentum_{}s".format(window)] = np.log(series / series.shift(window)).to_numpy()
    for window in range_windows:
        high = series.rolling(window, min_periods=window).max()
        low = series.rolling(window, min_periods=window).min()
        output["high_low_range_{}s".format(window)] = ((high - low) / series).to_numpy()
    return output

