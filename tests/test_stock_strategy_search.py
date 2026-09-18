from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.directional_factors import simulate_single_day
from research_engine.analysis.stock_strategy_search import (
    CALIBRATION, calibrate, fresh_gate, pick, signals_for, stock_observations,
)


def frame():
    return pd.DataFrame(dict(timestamp=pd.date_range('2026-09-15T13:30:00Z', periods=1801, freq='1s'),
                             qqq_bid=99.99, qqq_ask=100.01, qqq_depth_age_seconds=0., bid_size=100., ask_size=100.,
                             buy_volume_30s=100., sell_volume_30s=50., total_volume_5s=30.))


def test_future_prices_do_not_change_past_or_reveal_opening_range_early():
    source = frame()
    initial = stock_observations(source, StudySettings())
    assert initial[initial.seconds_from_open.lt(900)].opening_high.isna().all()
    source.loc[1000:, ['qqq_bid', 'qqq_ask']] += 20
    later = stock_observations(source, StudySettings())
    pd.testing.assert_frame_equal(initial[initial.seconds_from_open.lt(1000)], later[later.seconds_from_open.lt(1000)])
    assert later[later.seconds_from_open.ge(900)].opening_high.eq(100.).all()
    source.loc[700, 'qqq_depth_age_seconds'] = 100.
    bad = stock_observations(source, StudySettings())
    assert not bad[bad.seconds_from_open.between(700, 1000)].valid300.any()


def test_denser_sampling_preserves_original_windows_and_values():
    source = frame()
    source.loc[400:450, ['qqq_bid','qqq_ask']] += .5
    settings = StudySettings()
    original = stock_observations(source, settings)
    dense = stock_observations(source, settings, sampling_seconds=15)
    same = dense[dense.seconds_from_open.mod(30).eq(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, same)
    assert set(dense.seconds_from_open.mod(60)) == {0,15,30,45}
    with pytest.raises(ValueError, match='divisor'):
        stock_observations(source, settings, sampling_seconds=7)


def test_thresholds_exclude_current_date_and_current_price_extremes():
    settings = replace(StudySettings(), calibration_days=2, min_history_samples=2)
    current = stock_observations(frame(), settings)
    history = []
    for date in ['2026-09-11', '2026-09-14']:
        past = current.copy()
        past['date'] = date
        history.append(past)
    before = calibrate(current, history, settings)
    current.loc[:, 'r60'] = 100.
    after = calibrate(current, history, settings)
    assert before.threshold_r60.equals(after.threshold_r60)
    with pytest.raises(ValueError, match='earlier'):
        calibrate(current, [history[0], current], settings)


def test_continuation_and_reversal_are_predeclared_and_context_only_filters():
    current = stock_observations(frame(), StudySettings()).iloc[15:18].copy()
    current['ready'] = True
    current['valid300'] = True
    current['r60'] = [.02, -.02, 0.]
    for field in CALIBRATION:
        current['threshold_' + field] = .01
    current['past_vol'] = .005
    base = signals_for(current, 'all')
    assert base[base.factor.eq('momentum60')].direction.tolist() == [1, -1, 0]
    assert base[base.factor.eq('reversal60')].direction.tolist() == [-1, 1, 0]
    filtered = signals_for(current, 'high_vol')
    assert filtered.direction.eq(0).all()


def test_selection_rejects_profit_concentrated_in_one_day_or_too_few_trades():
    rows = pd.DataFrame(dict(factor=['a','b','c','d'], horizon=180, closed=[40,40,12,40],
                             active_days=5, positive_days=3, stress_net=[20.,20.,20.,-1.],
                             without_best_day=[1.,-1.,1.,1.]))
    assert pick(rows).factor.tolist() == ['a']


def test_entry_gate_rejection_does_not_occupy_position_or_change_exit_age_limit():
    t = pd.Timestamp('2026-09-15T13:35:30Z')
    source = pd.DataFrame(dict(timestamp=[t, t+pd.Timedelta(seconds=60)], date='2026-09-15',
                               factor='momentum60', scheduled=True, signal_status='signal', direction=1,
                               spot=100., regime='research', session_close=t+pd.Timedelta(hours=1)))
    events = pd.DataFrame([dict(symbol='CALL100',option_right='CALL',option_strike=100.,
                               available_at=t+pd.Timedelta(seconds=s),best_bid=b,best_ask=b+.02,
                               best_bid_size=1,best_ask_size=1) for s,b in [(-.5,1.),(61,1.),(119,1.1)]])
    books = ContractBooks(events)
    settings = replace(StudySettings(), holding_seconds=60)
    ledger = simulate_single_day(source, books, settings, entry_gate=fresh_gate(books))
    assert ledger.status.tolist() == ['entry_rejected','closed']
    assert ledger.iloc[1].exit_quote_age == 2.
    assert ledger.iloc[1].entry_quote_age == 0.
