from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.directional_factors import simulate_single_day
from research_engine.analysis.option_target_search import choose_signals, decision_quotes, episode_labels, forecasts
from research_engine.analysis.stock_prediction_search import FEATURES
from research_engine.analysis.stock_strategy_search import fresh_gate


def sample():
    t = pd.Timestamp('2026-09-15T14:00:00Z')
    current = pd.DataFrame(dict(timestamp=[t, t + pd.Timedelta(seconds=30)], date='2026-09-15',
                               half_hour=1, spot=100., past_vol=.001, scheduled=True, valid300=True,
                               session_close=t + pd.Timedelta(hours=1)))
    events = pd.DataFrame([dict(symbol='CALL100', option_right='CALL', option_strike=100.,
                               available_at=t + pd.Timedelta(seconds=s), best_bid=bid, best_ask=bid+.02,
                               best_bid_size=1, best_ask_size=1)
                           for s, bid in [(0,1.), (1,1.), (30,1.1), (31,1.1), (61,1.2), (91,1.3)]])
    return current, ContractBooks(events)


def test_independent_episode_labels_preserve_overlap_costs_missing_and_unclosed():
    current, books = sample()
    settings = StudySettings()
    quotes = decision_quotes(current, books, settings)
    labelled, episodes = episode_labels(current, quotes, books, settings)
    calls = episodes[episodes.right.eq('CALL') & episodes.horizon.eq(60)]
    assert calls.status.tolist() == ['closed', 'closed']
    assert calls.iloc[1].entry_time < calls.iloc[0].exit_time
    signals = current.assign(factor='x', regime='x', signal_status='signal', direction=1)
    portfolio = simulate_single_day(signals, books, replace(settings, holding_seconds=60), entry_gate=fresh_gate(books))
    assert portfolio.status.tolist() == ['closed', 'busy']
    assert labelled.iloc[0].target_CALL_60 == pytest.approx(portfolio.iloc[0].net_pnl)
    assert labelled.iloc[0].target_CALL_60 == pytest.approx((1.2*.9999 - 1.02*1.0001)*100 - 1.3)
    assert labelled.target_PUT_60.isna().all()
    unresolved = episodes[episodes.right.eq('CALL') & episodes.horizon.eq(300)]
    assert unresolved.status.eq('unclosed').all()
    np.testing.assert_allclose(unresolved.target_net, -unresolved.entry_debit)


def test_choices_use_available_right_and_fixed_buffer_without_future_labels():
    current, books = sample()
    quotes = decision_quotes(current, books, StudySettings())
    predictions = dict(CALL=np.array([2.,4.]), PUT=np.array([100.,100.]))
    current['target_CALL_60'] = -1e9
    signals = choose_signals(current, quotes, predictions, 'ridge', 3)
    assert signals.direction.tolist() == [0, 1]
    assert signals.forecast_net.tolist() == [2., 4.]
    assert signals.iloc[1].decision_symbol == 'CALL100'
    assert not any(col.startswith('target_') for col in signals)


def test_option_targets_use_only_previous_dates_and_correct_right():
    rng = np.random.RandomState(7)
    history = []
    for date in ['2026-08-31','2026-09-01','2026-09-02','2026-09-03','2026-09-04']:
        day = pd.DataFrame(rng.normal(size=(30, len(FEATURES))), columns=FEATURES)
        day['date'], day['valid300'] = date, True
        day['target_CALL_60'], day['target_PUT_60'] = 10., -10.
        history.append(day)
    current = history[0].iloc[:3].copy()
    current['date'] = '2026-09-08'
    before, details = forecasts(current, history, 60, 'ridge', min_samples=20)
    current['target_CALL_60'], current['target_PUT_60'] = -1e9, 1e9
    after, _ = forecasts(current, history, 60, 'ridge', min_samples=20)
    for right, value in [('CALL', 10.), ('PUT', -10.)]:
        np.testing.assert_allclose(before[right], value)
        np.testing.assert_array_equal(before[right], after[right])
    assert all(row['history_last_date'] == '2026-09-04' for row in details)
    with pytest.raises(ValueError, match='earlier'):
        forecasts(current, history[:4]+[current], 60, 'ridge', min_samples=20)
