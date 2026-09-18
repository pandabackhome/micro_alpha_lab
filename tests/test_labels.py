import numpy as np
import pandas as pd

from research_engine.labels.forward_returns import add_forward_labels


def test_forward_return_and_excursions():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="1s", tz="UTC"),
        "qqq_mid": [100.0, 101.0, 99.0, 102.0],
    })
    config = {
        "labels": {
            "forward_returns": {"horizons": [2]},
            "direction": {"horizons": [2], "threshold_bps": 5},
            "excursions": {"horizons": [2]},
        }
    }
    labels = add_forward_labels(frame, config)
    assert np.isclose(labels.loc[0, "future_ret_2s"], -0.01)
    assert labels.loc[0, "future_direction_2s"] == "DOWN"
    assert np.isclose(labels.loc[0, "mfe_2s"], 0.01)
    assert np.isclose(labels.loc[0, "mae_2s"], -0.01)
    assert np.isnan(labels.loc[2, "mfe_2s"])
    assert labels.loc[2, "future_direction_2s"] is None
