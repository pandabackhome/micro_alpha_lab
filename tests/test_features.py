import numpy as np
import pandas as pd

from research_engine.features.orderbook import depth_imbalance, microprice, ofi_series
from research_engine.features.price import price_features
from research_engine.features.snapshot import causal_asof
from research_engine.features.trade_flow import trade_flow_features
from research_engine.features.options import mapped_option_features


def grid():
    return pd.date_range("2026-09-15T13:30:00Z", periods=4, freq="1s")


def test_snapshot_never_uses_future_event():
    source_times = pd.to_datetime(["2026-09-15T13:30:00.100Z", "2026-09-15T13:30:01.100Z"])
    result = causal_asof(grid(), source_times, {"price": [100.0, 200.0]})["price"]
    assert np.isnan(result[0])
    assert result[1] == 100.0
    assert result[2] == 200.0


def test_snapshot_arrow_microsecond_precision_never_selects_future():
    source_times = pd.Series(pd.to_datetime([
        "2026-09-15T13:30:00.100Z", "2026-09-15T20:10:00.100Z",
    ]).as_unit("us"))
    result = causal_asof(grid(), source_times, {"price": [100.0, 900.0]})["price"]
    assert np.isnan(result[0])
    assert (result[1:] == 100.0).all()


def test_price_return_uses_only_past():
    original = np.array([100.0, 101.0, 102.0, 103.0])
    changed_future = np.array([100.0, 101.0, 999.0, 103.0])
    first = price_features(original, [1], [1], [2])["ret_1s"]
    second = price_features(changed_future, [1], [1], [2])["ret_1s"]
    assert first[1] == second[1]
    assert np.isclose(first[1], np.log(101 / 100))


def test_trade_imbalance_and_zero_denominator():
    times = pd.to_datetime(["2026-09-15T13:30:00Z"] * 3)
    values = trade_flow_features(
        grid(), times, [6, 2, 100],
        ["TradeDirection.Up", "TradeDirection.Down", "TradeDirection.Neutral"], [1],
    )
    assert values["trade_imbalance_1s"][0] == 0.5
    assert values["trade_imbalance_1s"][1] == 0.0
    assert values["total_volume_1s"][0] == 108


def test_depth_imbalance_and_microprice():
    assert depth_imbalance([30], [10])[0] == 0.5
    assert microprice([100], [101], [30], [10])[0] == 100.75


def test_ofi_formula():
    result = ofi_series([100, 100, 100.5], [10, 12, 5], [101, 101, 101], [10, 8, 8])
    assert np.isnan(result[0])
    assert result[1] == 4  # bid +2 and ask-size decrease +2
    assert result[2] == 5  # better bid contributes the new size; ask unchanged


def test_option_relative_strike_returns_do_not_cross_contracts():
    stamps = grid()
    quotes = pd.DataFrame({
        "symbol": ["QQQ260915C100000.US", "QQQ260915C101000.US"],
        "option_right": ["CALL", "CALL"], "option_strike": [100.0, 101.0],
        "available_at": [stamps[0], stamps[0]], "price": [1.0, 2.0],
    })
    depths = pd.DataFrame({
        "symbol": quotes["symbol"], "option_right": quotes["option_right"],
        "option_strike": quotes["option_strike"], "available_at": quotes["available_at"],
        "best_bid": [0.9, 1.9], "best_ask": [1.1, 2.1],
        "best_bid_size": [10.0, 10.0], "best_ask_size": [10.0, 10.0],
    })
    trades = pd.DataFrame(columns=["symbol", "option_right", "option_strike",
                                   "available_at", "volume", "direction"])
    result = mapped_option_features(stamps, np.array([100.0, 101.0, 101.0, 101.0]),
                                    quotes, depths, trades, [0], [1], [1, 5], 5, 10)
    assert result["call_atm_option_mid"][0] == 1.0
    assert result["call_atm_option_mid"][1] == 2.0
    assert np.isnan(result["call_atm_option_return_1s"][1])
