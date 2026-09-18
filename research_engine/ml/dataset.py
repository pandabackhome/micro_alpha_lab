from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import pandas as pd


LABEL_PREFIXES = ("future_", "mfe_", "mae_", "short_mfe_", "short_mae_")
NONFEATURES = {"timestamp", "session_bucket", "qqq_last", "qqq_bid", "qqq_ask", "qqq_mid"}


def feature_columns(frame: pd.DataFrame) -> List[str]:
    """Explicit allowlist by provenance; forward labels are never features."""
    return [name for name in frame.select_dtypes(include="number").columns
            if name not in NONFEATURES and not name.startswith(LABEL_PREFIXES)]


def load_dataset(feature_paths: Sequence[Union[str, Path]], label: str = "future_direction_30s",
                 stride: int = 5) -> pd.DataFrame:
    frames = []
    for path in sorted(map(Path, feature_paths)):
        labels = path.parent.parent / "labels" / path.name
        if not labels.exists():
            continue
        feature_frame = pd.read_parquet(path).iloc[::stride].reset_index(drop=True)
        label_frame = pd.read_parquet(labels).iloc[::stride].reset_index(drop=True)
        if len(feature_frame) != len(label_frame) or not feature_frame["timestamp"].equals(label_frame["timestamp"]):
            raise ValueError("feature/label timestamp mismatch: {}".format(path))
        needed = list(dict.fromkeys([label] + [name for name in label_frame.columns if name != "timestamp"]))
        frame = pd.concat([feature_frame, label_frame[needed]], axis=1)
        frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError("no matching feature/label files")
    return pd.concat(frames, ignore_index=True)
