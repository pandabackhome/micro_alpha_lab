from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer

from research_engine.config import portable_path


def add_forward_labels(frame: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add labels using future mid prices.

    Feature builders never call this function.  Keeping labels in a separate
    output file makes accidental training on forward calculations easier to
    audit and prevents a label from becoming a feature via a broad selector.
    """
    result = pd.DataFrame({"timestamp": frame["timestamp"]})
    label_config = config["labels"]
    entry = frame["qqq_mid"].astype(float)
    if "qqq_depth_age_seconds" in frame:
        # Do not silently label a minutes-old carried-forward mid as if it
        # were an observable executable price. Both endpoints and every
        # excursion point must come from a reasonably fresh book.
        entry = entry.where(frame["qqq_depth_age_seconds"] <= float(label_config.get("max_mid_age_seconds", 5)))
    for horizon in label_config["forward_returns"]["horizons"]:
        result["future_ret_{}s".format(horizon)] = entry.shift(-int(horizon)) / entry - 1.0

    # A fixed basis-point band asks every session the same question, but the
    # sessions are not the same size. Across the 18 recorded days, 5bp at 30s
    # falls past the 95th percentile of |return| on a quiet one (2026-09-17:
    # 1.8% of rows non-FLAT) and near the 88th on a wide one (2026-09-16:
    # 10.3%) -- a 5.7x swing in how often the label fires at all, decided by
    # the day rather than by anything a model could learn. Downstream that is
    # not cosmetic: it drove the per-fold count of above-threshold predictions
    # from 10 to 490, which leaves precision at a probability cut incomparable
    # from one fold to the next.
    #
    # Scaling the band by a realised-volatility feature holds that rate roughly
    # constant -- 1.3x spread over the same 18 days at vol_k=2.0. `vol_column`
    # must name a *causal* feature, computed from past bars only, so the
    # label's definition stays clear of the future even though its value is a
    # forward return; pointing it at a forward column would leak.
    #
    # Measured effect once the labels were rebuilt: the model's precision at
    # prob>0.6 fell from 0.103 to 0.046 and its lift over the base rate from
    # 2.7-10.6x to 0.0-2.7x. That is the point of the change rather than an
    # argument against it -- the old figure was flattered by a threshold that
    # was easier to cross on exactly the high-volatility sessions the model
    # already preferred, so it mixed direction skill with volatility timing.
    #
    # Off unless configured. With no `vol_column` this writes exactly the
    # labels it wrote before the option existed.
    threshold = float(label_config["direction"]["threshold_bps"]) / 10000.0
    vol_column = label_config["direction"].get("vol_column")
    if vol_column:
        if vol_column not in frame:
            raise KeyError(
                "labels.direction.vol_column={!r} is not in the feature frame; add it to "
                "the columns write_labels reads, or remove the setting".format(vol_column)
            )
        band = float(label_config["direction"].get("vol_k", 2.0)) * frame[vol_column].astype(float)
        # A missing or non-positive volatility defines no band. Those rows fall
        # back to the fixed threshold rather than calling every tick a move.
        band = band.where(band > 0, threshold)
    else:
        band = pd.Series(threshold, index=frame.index)

    for horizon in label_config["direction"].get("horizons", [label_config["direction"].get("horizon", 30)]):
        returns = entry.shift(-int(horizon)) / entry - 1.0
        direction = np.select([returns > band, returns < -band], ["UP", "DOWN"], default="FLAT").astype(object)
        direction[pd.isna(returns)] = None
        result["future_direction_{}s".format(horizon)] = direction

    for horizon in label_config["excursions"]["horizons"]:
        horizon = int(horizon)
        # At t, shifted_future[t:t+h] equals original prices t+1 ... t+h.
        indexer = FixedForwardWindowIndexer(window_size=horizon)
        shifted_future = entry.shift(-1)
        future_max = shifted_future.rolling(indexer, min_periods=horizon).max()
        future_min = shifted_future.rolling(indexer, min_periods=horizon).min()
        result["mfe_{}s".format(horizon)] = future_max / entry - 1.0
        result["mae_{}s".format(horizon)] = future_min / entry - 1.0
        result["short_mfe_{}s".format(horizon)] = 1.0 - future_min / entry
        result["short_mae_{}s".format(horizon)] = 1.0 - future_max / entry
    return result


def write_labels(feature_path: Union[str, Path], output: Union[str, Path], config: Dict) -> Dict:
    feature_path, output = Path(feature_path), Path(output)
    columns = ["timestamp", "qqq_mid", "qqq_depth_age_seconds"]
    # Read the volatility column only when the direction band is scaled by it;
    # the narrow column list is what keeps this function cheap on a 414-column
    # feature file, and a column nobody asked for should not widen it.
    vol_column = config["labels"]["direction"].get("vol_column")
    if vol_column and vol_column not in columns:
        columns.append(vol_column)
    frame = pd.read_parquet(feature_path, columns=columns)
    labels = add_forward_labels(frame, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output, index=False, compression="zstd")
    result = {"output": str(output), "rows": len(labels), "columns": len(labels.columns)}
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(dict(result, output=portable_path(output)), indent=2) + "\n")
    return result
