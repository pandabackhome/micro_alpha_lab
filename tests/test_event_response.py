from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.directional_factors import simulate_single_day
from research_engine.analysis.event_response import (
    RULES, calibrate, estimated_cost, latency_attribution, latency_diagnostics, make_signals,
    observed_events, option_observations, paired_comparisons, summarize,
)


T = pd.Timestamp("2026-09-15T13:30:30Z")


def observation():
    row = dict(timestamp=T, date="2026-09-15", half_hour=0, spot=100., past_vol=.001,
               session_close=pd.Timestamp("2026-09-15T20:00:00Z"), scheduled=True,
               regime="high", pressure=3., previous_pressure=2., pressure_q75=1.,
               spot_change_5s=.1, spot_before_5s=99.9, spot_age=0., spot_before_age=0.,
               spot_spread=.02, previous_high=99.99, previous_low=99.8, option_flow=.8,
               flow_q75=.5, impulse_q75=.05)
    for right, beta in (("call", .5), ("put", -.5)):
        row.update({right + "_beta": beta, right + "_response_status": "ok",
                    right + "_symbol": right.upper() + "100.0", right + "_change_5s": 0.,
                    right + "_cost_estimate": .033202, right + "_mid_before": 1.,
                    right + "_mid_now": 1., right + "_spread": .02})
    return row


def books(updates, right="CALL", strike=100.):
    return pd.DataFrame([dict(symbol=right + str(strike), option_right=right, option_strike=strike,
                              available_at=T + pd.Timedelta(seconds=s), best_bid=bid, best_ask=ask,
                              best_bid_size=1, best_ask_size=1) for s, bid, ask in updates])


def spot_frame():
    stamps = pd.date_range(T - pd.Timedelta(seconds=30), periods=121, freq="1s")
    return pd.DataFrame(dict(timestamp=stamps, qqq_bid=99.99, qqq_ask=100.01, qqq_depth_age_seconds=0.,
                             bid_size=100., ask_size=100., buy_volume_5s=50., sell_volume_5s=10.,
                             call_buy_volume=40., call_sell_volume=10., put_buy_volume=5., put_sell_volume=15.))


def test_event_features_use_completed_past_range_and_normalized_pressure():
    frame = spot_frame()
    original = observed_events(frame, StudySettings())
    assert original.iloc[0].pressure == .2
    assert original.iloc[0].previous_pressure == .2
    assert original.iloc[0].option_flow == pytest.approx(40 / 70)
    frame.loc[31:, ["qqq_bid", "qqq_ask", "buy_volume_5s", "call_buy_volume"]] *= 2
    changed = observed_events(frame, StudySettings())
    pd.testing.assert_frame_equal(original.iloc[:1], changed.iloc[:1])
    frame.loc[30, ["qqq_bid", "qqq_ask"]] += 1
    event = observed_events(frame, StudySettings()).iloc[0]
    assert event.spot == 101. and event.previous_high == 100.
    frame.loc[28, "bid_size"] = 0
    assert pd.isna(observed_events(frame, StudySettings()).iloc[0].pressure)
    frame.loc[:, ["call_buy_volume", "call_sell_volume", "put_buy_volume", "put_sell_volume"]] = 0
    assert pd.isna(observed_events(frame, StudySettings()).iloc[0].option_flow)


def test_history_only_calibrates_thresholds_and_response_coefficient():
    settings = replace(StudySettings(), calibration_days=2, min_history_samples=20)
    history = []
    for date in ("2026-09-11", "2026-09-14"):
        rows = []
        for i in range(20):
            row = observation()
            row.update(date=date, pressure=float(i), spot_change_5s=(i + 1) * .01,
                       call_change_5s=(i + 1) * .004, put_change_5s=-(i + 1) * .006)
            rows.append(row)
        history.append(pd.DataFrame(rows))
    current = pd.DataFrame([observation()])
    before = calibrate(current, history, settings)
    assert before.iloc[0].call_beta == pytest.approx(.4)
    assert before.iloc[0].put_beta == pytest.approx(-.6)
    assert before.iloc[0].call_beta_samples == 40
    assert before.iloc[0].pressure_q75 == pytest.approx(14.25)
    current.loc[0, ["pressure", "call_change_5s", "spot_change_5s"]] = 1e6
    after = calibrate(current, history, settings)
    pd.testing.assert_frame_equal(before[["call_beta", "put_beta", "pressure_q75"]], after[["call_beta", "put_beta", "pressure_q75"]])
    assert before.iloc[0].history_last_date == "2026-09-14"
    with pytest.raises(ValueError, match="earlier"):
        calibrate(current, [history[0], current], settings)
    for day in history:
        day["call_response_status"] = "stale_quote"
    assert pd.isna(calibrate(current, history, settings).iloc[0].call_beta)


@pytest.mark.parametrize("sign", [1, -1])
def test_breakout_and_exhaustion_have_fixed_symmetric_directions(sign):
    row = observation()
    row.update(pressure=3. * sign, previous_pressure=2. * sign,
               option_flow=.8 * sign, previous_low=100.01 if sign < 0 else 99.8)
    result = make_signals(pd.DataFrame([row])).set_index("factor")
    assert result.loc["pressure_breakout", "direction"] == sign
    assert result.loc["breakout_confirmed", "direction"] == sign
    assert result.loc["pressure_exhaustion", "direction"] == 0
    row.update(pressure=.5 * sign, spot_change_5s=.005, option_flow=-.8 * sign)
    result = make_signals(pd.DataFrame([row])).set_index("factor")
    assert result.loc["pressure_exhaustion", "direction"] == -sign
    assert result.loc["exhaustion_confirmed", "direction"] == -sign
    row["option_flow"] *= -1
    result = make_signals(pd.DataFrame([row])).set_index("factor")
    assert result.loc["exhaustion_confirmed", "signal_status"] == "unconfirmed"
    assert result.loc["pressure_exhaustion", "direction"] == -sign


def test_response_uses_same_contract_and_rejects_stale_past_quotes():
    row = observation()
    events = books([(-5, .99, 1.01), (0, 1., 1.02)])
    events["available_at"] = events.available_at.dt.as_unit("us")
    result = option_observations(pd.DataFrame([row]), ContractBooks(events), StudySettings()).iloc[0]
    assert result.call_response_status == "ok"
    assert result.call_change_5s == pytest.approx(.01)
    assert result.put_response_status == "missing_contract"
    assert result.call_cost_estimate == pytest.approx(.033202)
    stale = books([(-7, .99, 1.01), (0, 1., 1.02)])
    assert "stale_quote" in option_observations(pd.DataFrame([row]), ContractBooks(stale), StudySettings()).iloc[0].call_response_status
    # A newly observed, nearer strike cannot borrow the old strike's past quote.
    row["spot"] = 100.1
    mixed = ContractBooks(pd.concat([events, books([(0, 20., 21.)], strike=100.1)]))
    result = option_observations(pd.DataFrame([row]), mixed, StudySettings()).iloc[0]
    assert result.call_symbol == "CALL100.1"
    assert "missing_quote" in result.call_response_status
    assert pd.isna(result.call_change_5s)


def test_gap_cost_gate_and_one_second_diagnostic_do_not_change_signals():
    row = observation()
    row["call_cost_estimate"] = estimated_cost(dict(bid=.99, ask=1.01), StudySettings())
    panel = make_signals(pd.DataFrame([row]))
    gap = panel[panel.factor.eq("option_response_gap")].iloc[0]
    assert gap.signal_status == "signal" and gap.response_gap == pytest.approx(.05)
    original = panel.copy(deep=True)
    frame = spot_frame()
    frame.loc[:, "qqq_bid"] = 99.99
    frame.loc[:, "qqq_ask"] = 100.01
    # At t+1 the option has already moved by $0.10, removing the residual.
    result = latency_diagnostics(panel, frame, ContractBooks(books([(1, 1.09, 1.11)])), StudySettings())
    assert result.iloc[0].status == "disappears"
    pd.testing.assert_frame_equal(panel, original)
    row["call_cost_estimate"] = row["call_beta"] * row["spot_change_5s"]
    result = make_signals(pd.DataFrame([row])).set_index("factor")
    assert result.loc["spot_impulse", "signal_status"] == "signal"
    assert result.loc["option_response_gap", "signal_status"] == "no_signal"


def test_three_minute_positions_block_overlaps_and_reject_late_entries():
    row = observation()
    source = make_signals(pd.DataFrame([row]))
    source = source[source.factor.eq("pressure_only")]
    copies = []
    for seconds in (0, 60, 240):
        copy = source.copy()
        copy["timestamp"] = T + pd.Timedelta(seconds=seconds)
        copies.append(copy)
    events = books([(0, 1., 1.02), (1, 1., 1.02), (181, 1.1, 1.12),
                     (240, 1., 1.02), (241, 1., 1.02), (421, 1.1, 1.12)])
    settings = replace(StudySettings(), holding_seconds=180)
    result = simulate_single_day(pd.concat(copies), ContractBooks(events), settings)
    assert result.status.tolist() == ["closed", "busy", "closed"]
    assert result[result.status.eq("closed")].holding_seconds.eq(180).all()
    source["session_close"] = T + pd.Timedelta(seconds=180)
    assert simulate_single_day(source, ContractBooks(events), settings).iloc[0].status == "session_end"


def test_empty_signal_groups_remain_in_summaries_without_inventing_pnl():
    row = observation()
    row.update(pressure=0., previous_pressure=0., spot_change_5s=0.)
    signals = make_signals(pd.DataFrame([row]))
    panels, ledgers = [], []
    for horizon in (30, 180):
        panel = signals.copy()
        panel["horizon"] = horizon
        panel["future_return_bps"] = 0.
        panel["delayed_return_bps"] = 0.
        panels.append(panel)
        for rule in RULES:
            ledger = simulate_single_day(signals[signals.factor.eq(rule)], None, StudySettings())
            ledger["horizon"] = horizon
            ledgers.append(ledger)
    execution, direction, daily, _ = summarize(pd.concat(panels), pd.concat(ledgers), [row["date"]])
    assert len(execution) == 14 and execution.closed.eq(0).all()
    assert execution.mean_net_pnl.isna().all() and direction.mean_signed_bps.isna().all()
    comparison, sensitivity = paired_comparisons(daily)
    assert comparison[comparison.metric.eq("mean_net_pnl")].paired_days.eq(0).all()
    assert sensitivity.empty
    assert latency_attribution(pd.concat(ledgers), pd.DataFrame()).empty
