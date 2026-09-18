from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research_engine.features.pipeline import build_feature_frame
from research_engine.ingest.normalize import SCHEMA, normalize_payload
from research_engine.labels.forward_returns import add_forward_labels


def test_entire_pipeline_only_uses_events_available_by_snapshot(tmp_path):
    t0 = "2026-09-15T13:30:00Z"
    t2 = "2026-09-15T13:30:02Z"
    option = "QQQ260915C100000.US"
    payloads = [
        {"kind": "quote", "symbol": "QQQ.US", "ts": t0, "received_at": t0, "price": 100.0},
        {"kind": "depth", "symbol": "QQQ.US", "received_at": t0,
         "bids": [[1, 99.9, 10, 0], [2, 99.8, 5, 0]],
         "asks": [[1, 100.1, 10, 0], [2, 100.2, 20, 0]]},
        {"kind": "quote", "symbol": option, "ts": t0, "received_at": t0, "price": 1.0},
        {"kind": "depth", "symbol": option, "received_at": t0,
         "bids": [[1, 0.9, 10, 0]], "asks": [[1, 1.1, 10, 0]]},
        # Historical exchange timestamp but not *received* until t2.
        {"kind": "trade", "symbol": "QQQ.US", "ts": t0, "received_at": t2,
         "price": 100.1, "volume": 8, "direction": "TradeDirection.Up"},
        {"kind": "trade", "symbol": option, "ts": t0, "received_at": t2,
         "price": 1.1, "volume": 5, "direction": "TradeDirection.Up"},
        {"kind": "quote", "symbol": "QQQ.US", "ts": t2, "received_at": t2, "price": 100.2},
        {"kind": "depth", "symbol": "QQQ.US", "received_at": t2,
         "bids": [[1, 100.1, 100, 0]], "asks": [[1, 100.3, 10, 0]]},
        {"kind": "quote", "symbol": option, "ts": t2, "received_at": t2, "price": 1.2},
        {"kind": "depth", "symbol": option, "received_at": t2,
         "bids": [[1, 1.1, 100, 0]], "asks": [[1, 1.3, 10, 0]]},
    ]
    config = {
        "underlying": "QQQ.US", "timezone": "America/New_York",
        "session": {"open": "09:30:00", "close": "09:30:03"}, "snapshot_interval": "1s",
        "features": {
            "returns": {"windows": [1]}, "momentum": {"windows": [1]},
            "ranges": {"windows": [2]}, "trade_flow": {"windows": [1, 5]},
            "orderbook": {"windows": [1], "levels": [1, 5, 10]}, "volatility": {"windows": [2]},
        },
        "options": {"relative_strikes": [0], "return_windows": [1], "volume_windows": [1, 5],
                    "flow_window": 1, "volume_burst_lookback": 10, "near_atm_distance": 0.003},
        "labels": {"forward_returns": {"horizons": [2]},
                   "direction": {"horizons": [2], "threshold_bps": 5},
                   "excursions": {"horizons": [2]}},
    }
    path = tmp_path / "events.parquet"
    pq.write_table(pa.Table.from_pylist([normalize_payload(i + 1, p) for i, p in enumerate(payloads)], schema=SCHEMA), path)
    frame = build_feature_frame(path, date(2026, 9, 15), config)
    assert frame.loc[1, "qqq_mid"] == 100.0
    assert frame.loc[1, "call_atm_option_mid"] == 1.0
    assert frame.loc[1, "trade_count_1s"] == 0
    assert frame.loc[1, "call_atm_option_volume_1s"] == 0
    assert frame.loc[1, "ret_1s"] == 0
    assert frame.loc[1, "bid_size_l5"] == 15
    assert frame.loc[1, "ask_size_l5"] == 30
    assert np.isclose(frame.loc[1, "depth_imbalance_l5"], -1 / 3)
    assert np.isclose(frame.loc[2, "qqq_mid"], 100.2)
    assert np.isclose(frame.loc[2, "call_atm_option_mid"], 1.2)
    assert frame.loc[2, "trade_count_1s"] == 1
    assert frame.loc[2, "call_atm_option_volume_1s"] == 5
    labels = add_forward_labels(frame, config)
    assert np.isclose(labels.loc[0, "future_ret_2s"], 0.002)
    assert np.isclose(labels.loc[0, "mfe_2s"], 0.002)
