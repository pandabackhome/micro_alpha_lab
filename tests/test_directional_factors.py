from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.directional_factors import (
    FACTORS, direction_metrics, execution_metrics, factor_observations,
    factor_signals, future_outcomes, simulate_single_day,
)


T = pd.Timestamp("2026-09-15T13:30:30Z")


def signals(seconds=(0,)):
    return pd.DataFrame({
        "timestamp": [T + pd.Timedelta(seconds=s) for s in seconds],
        "date": "2026-09-15", "half_hour": 0, "spot": 100.0,
        "past_vol": 0.001, "scheduled": True, "regime": "high",
        "session_close": pd.Timestamp("2026-09-15T20:00:00Z"),
        "factor": "ofi_5s", "direction": 1, "signal_status": "signal",
    })


def events(updates, right="CALL", strike=100.0):
    return pd.DataFrame([
        {"symbol": right + str(strike), "option_right": right, "option_strike": strike,
         "available_at": T + pd.Timedelta(seconds=seconds), "best_bid": bid,
         "best_ask": ask, "best_bid_size": 1, "best_ask_size": 1}
        for seconds, bid, ask in updates
    ])


def test_thresholds_only_use_previous_dates_and_sign_guard_preserves_ties():
    settings = replace(StudySettings(), calibration_days=2, min_history_samples=2)
    history = []
    for date in ("2026-09-11", "2026-09-14"):
        day = pd.DataFrame({"date": date, "half_hour": [0] * 4, "past_vol": [0.001] * 4})
        for factor in FACTORS:
            day[factor] = [1., 2., 3., 4.]
        history.append(day)
    current = signals((0, 60, 120, 180, 240)).drop(columns=["factor", "direction", "signal_status"])
    for factor in FACTORS:
        current[factor] = [0., 1., 3.25, 4., -1.]
    result = factor_signals(current, history, settings)
    one = result[result.factor.eq("ofi_5s")]
    assert one.direction.tolist() == [0, 0, 0, 1, -1]
    assert one.factor_q25.eq(1.75).all() and one.factor_q75.eq(3.25).all()
    assert one.history_last_date.eq("2026-09-14").all()
    current.loc[4, list(FACTORS)] = 1e9
    changed = factor_signals(current, history, settings)
    for factor in FACTORS:
        pd.testing.assert_frame_equal(result[result.factor.eq(factor)].iloc[:4],
                                      changed[changed.factor.eq(factor)].iloc[:4])
    with pytest.raises(ValueError, match="earlier"):
        factor_signals(current, [history[0], current], settings)


def test_future_prices_change_labels_but_not_observed_prefix():
    stamps = pd.date_range("2026-09-15T13:30:00Z", periods=121, freq="1s")
    frame = pd.DataFrame({"timestamp": stamps, "qqq_bid": 99.99, "qqq_ask": 100.01,
                          "qqq_depth_age_seconds": 0., "ofi_5s": 1., "buy_volume_5s": 5.,
                          "sell_volume_5s": 2., "depth_imbalance": 0.1})
    settings = StudySettings()
    original = factor_observations(frame, settings)
    labels = future_outcomes(frame, settings)
    frame.loc[31:, ["qqq_bid", "qqq_ask"]] += 1
    changed = factor_observations(frame, settings)
    pd.testing.assert_frame_equal(original.iloc[:1], changed.iloc[:1])
    assert original.iloc[0].signed_volume_5s == 3
    assert labels.loc[30, "future_return_bps"] == 0
    assert future_outcomes(frame, settings).loc[30, "future_return_bps"] == pytest.approx(100)
    frame.loc[60, "qqq_depth_age_seconds"] = 99
    assert pd.isna(future_outcomes(frame, settings).loc[30, "future_return_bps"])


@pytest.mark.parametrize("right,direction", [("CALL", 1), ("PUT", -1)])
def test_single_leg_fixed_contract_microseconds_latency_and_full_costs(right, direction):
    book = events([(0, 0.9, 0.92), (1, 1., 1.02), (31, 1.09, 1.11)], right)
    book["available_at"] = book.available_at.dt.as_unit("us")
    later = events([(2, 20., 21.)], right, 100.1)
    books = ContractBooks(pd.concat([book, later]))
    source = signals()
    source["direction"] = direction
    source["spot"] = 100.1
    row = simulate_single_day(source, books, StudySettings()).iloc[0]
    assert row.status == "closed" and row.symbol == right + "100.0"
    assert row.entry_time == T + pd.Timedelta(seconds=1)
    assert row.mid_pnl == pytest.approx(9.)
    assert row.spread_cost == pytest.approx(2.)
    assert row.slippage_cost == pytest.approx(0.0211)
    assert row.commission == 1.3
    assert row.net_pnl == pytest.approx(5.6789)
    assert row.net_pnl == pytest.approx((row.exit_fill - row.entry_fill) * 100 - 1.3)
    assert books.select_leg(T - pd.Timedelta(microseconds=1), 100., right, .5) is None


def test_filter_runs_independently_and_unclosed_position_blocks_all_day():
    # The second opportunity is viable, but the unclosed first position blocks it.
    books = ContractBooks(events([(0, 1., 1.02), (1, 1., 1.02),
                                 (120, 1., 1.02), (121, 1., 1.02), (151, 1.1, 1.12)]))
    source = signals((0, 120))
    source.loc[0, "regime"] = "low"
    all_rows = simulate_single_day(source, books, StudySettings(), "all")
    high_rows = simulate_single_day(source, books, StudySettings(), "high_only")
    assert all_rows.status.tolist() == ["unclosed", "busy"]
    assert high_rows.status.tolist() == ["filtered_out", "closed"]
    metrics = execution_metrics(all_rows)
    assert metrics["unclosed"] == 1 and metrics["closed"] == 0
    assert metrics["realized_net_pnl"] == 0
    assert metrics["net_with_unclosed_zero_recovery"] == pytest.approx(-102.6602)


def test_missing_entry_retains_candidate_and_low_price_delayed_exit_is_allowed():
    source = signals()
    invalid = ContractBooks(events([(0, np.nan, 1.02)]))
    row = simulate_single_day(source, invalid, StudySettings()).iloc[0]
    assert row.status == "entry_unavailable" and row.reason == "incomplete_quote"
    books = ContractBooks(events([(0, 1., 1.02), (1, 1., 1.02), (33, .01, .02)]))
    row = simulate_single_day(source, books, StudySettings()).iloc[0]
    assert row.status == "closed" and row.exit_delay_seconds == 2
    assert row.missing_exit_checks == 2 and row.net_pnl < -100


def test_direction_denominators_include_flats_and_audit_missing_labels():
    panel = pd.DataFrame({"signal_status": ["signal"] * 4 + ["no_signal"],
                          "factor_value": [1., -1., 2., 3., 0.], "direction": [1, -1, 1, 1, 0],
                          "future_return_bps": [2., -1., 0., np.nan, 99.],
                          "delayed_return_bps": [1., -2., 0., np.nan, 99.]})
    result = direction_metrics(panel)
    assert result["signals"] == 4 and result["labelled_signals"] == 3
    assert result["missing_future_labels"] == 1
    assert result["hit_rate"] == pytest.approx(2 / 3)
    assert result["random_sign_hit_baseline"] == pytest.approx(1 / 3)
    assert result["mean_signed_return_bps"] == 1
