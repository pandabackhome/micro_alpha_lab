from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import (
    BOOK_COLUMNS, ContractBooks, StudySettings, assign_regimes, observations,
    simulate_day, summarize,
)


T = pd.Timestamp("2026-09-15T13:30:30Z")


def signal_frame(seconds=(0,), regime="high"):
    return pd.DataFrame({
        "timestamp": [T + pd.Timedelta(seconds=x) for x in seconds],
        "date": "2026-09-15", "half_hour": 0, "spot": 100.0,
        "past_vol": 0.001, "scheduled": True, "regime": regime,
        "session_close": pd.Timestamp("2026-09-15T20:00:00Z"),
    })


def book_events(updates, strike=100.0):
    """Each update is (seconds, call_bid, call_ask, put_bid, put_ask)."""
    rows = []
    for seconds, cb, ca, pb, pa in updates:
        for right, bid, ask in (("CALL", cb, ca), ("PUT", pb, pa)):
            rows.append({"symbol": right + str(strike), "option_right": right,
                         "option_strike": strike, "available_at": T + pd.Timedelta(seconds=seconds),
                         "best_bid": bid, "best_ask": ask, "best_bid_size": 1, "best_ask_size": 1})
    return pd.DataFrame(rows, columns=BOOK_COLUMNS)


def test_volatility_prefix_does_not_change_when_future_prices_change():
    stamps = pd.date_range("2026-09-15T13:30:00Z", periods=121, freq="1s")
    mid = 100 + np.arange(121) * 0.001
    frame = pd.DataFrame({"timestamp": stamps, "qqq_bid": mid - 0.005,
                          "qqq_ask": mid + 0.005, "qqq_depth_age_seconds": 0.0})
    original = observations(frame, StudySettings())
    microsecond_grid = frame.copy()
    microsecond_grid["timestamp"] = microsecond_grid["timestamp"].dt.as_unit("us")
    pd.testing.assert_frame_equal(original, observations(microsecond_grid, StudySettings()))
    frame.loc[61:, ["qqq_bid", "qqq_ask"]] += 50
    changed = observations(frame, StudySettings())
    pd.testing.assert_frame_equal(original.iloc[:2], changed.iloc[:2])
    frame.loc[45, "qqq_depth_age_seconds"] = 100
    bad = observations(frame, StudySettings())
    assert pd.isna(bad.iloc[1]["past_vol"])


def test_regimes_use_only_previous_dates_and_preserve_ties():
    settings = replace(StudySettings(), calibration_days=2, min_history_samples=2)
    history = [pd.DataFrame({"date": day, "half_hour": [0, 0, 0, 0],
                             "past_vol": [1.0, 2.0, 3.0, 4.0]})
               for day in ("2026-09-11", "2026-09-14")]
    current = signal_frame((0, 60, 120))
    current["past_vol"] = [1.0, 2.5, 4.0]
    result = assign_regimes(current, history, settings)
    assert result["regime"].tolist() == ["low", "middle", "high"]
    assert result["q25"].tolist() == [1.75] * 3
    assert result["q75"].tolist() == [3.25] * 3
    current.loc[2, "past_vol"] = 1e9
    pd.testing.assert_frame_equal(result.iloc[:2], assign_regimes(current, history, settings).iloc[:2])
    assert result["history_last_date"].eq("2026-09-14").all()
    with pytest.raises(ValueError, match="earlier"):
        assign_regimes(current, [history[0], current], settings)
    for day in history:
        day["past_vol"] = 1.0
    current["past_vol"] = 1.0
    assert assign_regimes(current, history, settings)["regime"].eq("middle").all()


def test_contract_universe_and_quotes_do_not_look_ahead_or_hide_invalid_updates():
    old = book_events([(0, 1.0, 1.02, 1.0, 1.02), (2, np.nan, 1.02, 1.0, 1.02)])
    future = book_events([(10, 3, 3.02, 3, 3.02)], strike=100.1)
    books = ContractBooks(pd.concat([old, future]))
    assert books.select(T, 100.1, 0.5)[0] == 100.0
    assert books.quote("CALL100.1", T, 5)[1] == "missing_quote"
    assert books.quote("CALL100.0", T + pd.Timedelta(seconds=2), 5)[1] == "incomplete_quote"
    old.loc[0, "best_bid_size"] = 0
    assert ContractBooks(old).quote("CALL100.0", T, 5)[1] == "insufficient_size"


def test_arrow_microsecond_timestamps_have_same_availability_as_nanoseconds():
    events = book_events([(0, 1, 1.02, 1, 1.02), (1, 2, 2.02, 2, 2.02)])
    events["available_at"] = events["available_at"].dt.as_unit("us")
    books = ContractBooks(events)
    quote, reason = books.quote("CALL100.0", T + pd.Timedelta(milliseconds=500), 5)
    assert reason == "ok"
    assert quote["bid"] == 1 and quote["age"] == 0.5
    assert books.select(T - pd.Timedelta(microseconds=1), 100.0, 0.5) is None


def test_fixed_contract_delayed_entry_and_full_cost_accounting():
    settings = StudySettings()
    old = book_events([(0, 1.0, 1.01, 1.0, 1.01), (1, 1.0, 1.02, 1.0, 1.02),
                       (31, 1.09, 1.11, 0.89, 0.91)])
    new_atm = book_events([(0, 1.0, 1.02, 1.0, 1.02), (31, 20, 21, 20, 21)], strike=101.0)
    ledger = simulate_day(signal_frame(), ContractBooks(pd.concat([old, new_atm])), settings)
    row = ledger.iloc[0]
    assert row.status == "closed"
    assert row.call_symbol == "CALL100.0" and row.put_symbol == "PUT100.0"
    assert row.entry_time == T + pd.Timedelta(seconds=1)
    assert row.entry_ask == 2.04
    assert row.exit_bid == pytest.approx(1.98)
    assert row.mid_pnl == pytest.approx(-2)
    assert row.spread_cost == pytest.approx(4)
    assert row.slippage_cost == pytest.approx(0.0402)
    assert row.commission == pytest.approx(2.6)
    assert row.net_pnl == pytest.approx(-8.6402)
    assert row.net_pnl == pytest.approx((row.exit_fill - row.entry_fill) * 100 - 2.6)


def test_exit_waits_for_actual_fresh_quote_and_accepts_low_exit_price():
    events = book_events([(0, 1, 1.02, 1, 1.02), (1, 1, 1.02, 1, 1.02),
                          (33, 0.01, 0.02, 0.01, 0.02)])
    row = simulate_day(signal_frame(), ContractBooks(events), StudySettings()).iloc[0]
    assert row.status == "closed"
    assert row.exit_time == T + pd.Timedelta(seconds=33)
    assert row.exit_delay_seconds == 2
    assert row.missing_exit_checks == 2
    assert row.exit_bid == 0.02
    assert row.net_pnl < -200


def test_unclosed_position_is_retained_blocks_later_signals_and_has_stress_debit():
    events = book_events([(0, 1, 1.02, 1, 1.02), (1, 1, 1.02, 1, 1.02)])
    ledger = simulate_day(signal_frame((0, 60, 120)), ContractBooks(events), StudySettings())
    assert ledger["status"].tolist() == ["unclosed", "busy", "busy"]
    assert ledger.loc[0, "missing_exit_checks"] == 31
    assert "net_pnl" not in ledger.columns
    summary, _, _ = summarize(ledger)
    high = summary.set_index("regime").loc["high"]
    assert high.unclosed == 1 and high.closed == 0 and high.busy == 2
    assert high.unclosed_debit > 200
    assert high.net_with_unclosed_zero_recovery == -high.unclosed_debit
    assert pd.isna(high.mean_net_pnl)


def test_unavailable_entry_is_counted_without_dropping_candidate():
    events = book_events([(0, np.nan, 1.02, 1, 1.02)])
    ledger = simulate_day(signal_frame(), ContractBooks(events), StudySettings())
    assert len(ledger) == 1
    assert ledger.loc[0, "status"] == "entry_unavailable"
    assert "incomplete_quote" in ledger.loc[0, "reason"]
