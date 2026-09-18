import pandas as pd

from research_engine.backtest.engine import backtest_day


def test_entry_is_next_ask_exit_is_bid():
    features = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-15T13:30:00Z", periods=5, freq="1s"),
        "qqq_bid": [100, 100, 100.1, 100.2, 100.3],
        "qqq_ask": [100.2, 100.2, 100.3, 100.4, 100.5],
    })
    signals = pd.DataFrame({"timestamp": [features.loc[0, "timestamp"]], "side": ["LONG"]})
    config = {"execution": {"mode": "underlying", "slippage_bps": 0},
              "exit": {"max_holding_seconds": 2, "stop_loss_pct": -1, "take_profit_pct": 1}}
    trades = backtest_day(features, signals, config)
    assert len(trades) == 1
    assert trades.loc[0, "entry_price"] == 100.2
    assert trades.loc[0, "exit_price"] == 100.2
    assert trades.loc[0, "entry_time"] == features.loc[1, "timestamp"]


def test_exit_waits_for_next_fresh_bid_instead_of_reusing_old_one():
    features = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-15T13:30:00Z", periods=6, freq="1s"),
        "qqq_bid": [100, 100, 100, 100, 101, 102],
        "qqq_ask": [100.2, 100.2, 100.2, 100.2, 101.2, 102.2],
        "qqq_depth_age_seconds": [0, 0, 0, 10, 0, 0],
    })
    signals = pd.DataFrame({"timestamp": [features.loc[0, "timestamp"]], "side": ["LONG"]})
    config = {"execution": {"mode": "underlying", "slippage_bps": 0, "max_quote_age_seconds": 5},
              "exit": {"max_holding_seconds": 2, "max_exit_delay_seconds": 2,
                       "stop_loss_pct": -1, "take_profit_pct": 1}}
    trades = backtest_day(features, signals, config)
    assert trades.loc[0, "exit_time"] == features.loc[4, "timestamp"]
    assert trades.loc[0, "exit_price"] == 101


def test_signal_too_close_to_session_end_cannot_open_position():
    features = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-15T19:59:58Z", periods=3, freq="1s"),
        "qqq_bid": [100.0] * 3, "qqq_ask": [100.1] * 3,
    })
    signals = pd.DataFrame({"timestamp": [features.loc[0, "timestamp"]], "side": ["LONG"]})
    config = {"execution": {"mode": "underlying", "slippage_bps": 0},
              "exit": {"max_holding_seconds": 180, "stop_loss_pct": -1, "take_profit_pct": 1}}
    trades = backtest_day(features, signals, config)
    assert trades.empty
    assert trades.attrs["skipped_late_entries"] == 1
