import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.stock_prediction_search import FEATURES, cost_signals, forecast


def test_forecast_ignores_current_future_targets_and_rejects_current_date_history():
    rng = np.random.RandomState(7)
    history = []
    for date in ['2026-08-31','2026-09-01','2026-09-02','2026-09-03','2026-09-04']:
        day = pd.DataFrame(rng.normal(size=(30,len(FEATURES))), columns=FEATURES)
        day['date'], day['valid300'] = date, True
        day['target_60'] = day.r60 * 2
        history.append(day)
    current = history[0].iloc[:3].copy()
    current['date'] = '2026-09-08'
    before, details = forecast(current, history, 60, 'ridge', min_samples=20)
    current['target_60'] = 1e9
    after, _ = forecast(current, history, 60, 'ridge', min_samples=20)
    np.testing.assert_array_equal(before, after)
    assert details['training_samples'] == 150 and details['history_last_date'] == '2026-09-04'
    with pytest.raises(ValueError, match='earlier'):
        forecast(current, history[:4]+[current], 60, 'ridge', min_samples=20)


def test_cost_margin_uses_signal_time_quote_and_does_not_emit_training_labels():
    t = pd.Timestamp('2026-09-15T14:00:00Z')
    current = pd.DataFrame(dict(timestamp=[t]*3, date='2026-09-15', half_hour=1, spot=100., past_vol=.001,
                               session_close=t+pd.Timedelta(hours=1), scheduled=True, valid300=True, target_60=1000.))
    events = pd.DataFrame(dict(symbol=['CALL100','CALL100'], option_right='CALL', option_strike=100.,
                              available_at=[t,t+pd.Timedelta(seconds=1)], best_bid=[1.,20.], best_ask=[1.02,21.],
                              best_bid_size=1, best_ask_size=1))
    books = ContractBooks(events)
    details = dict(history_last_date='2026-09-14', training_samples=1000)
    basic = cost_signals(current, [2.,10.,20.], books, StudySettings(), 'ridge', 1, details)
    strict = cost_signals(current, [2.,10.,20.], books, StudySettings(), 'ridge', 2, details)
    assert basic.signal_status.tolist() == ['no_signal','signal','signal']
    assert strict.signal_status.tolist() == ['no_signal','no_signal','signal']
    assert 'target_60' not in basic.columns
    assert basic.cost_estimate.eq(basic.iloc[0].cost_estimate).all()
    assert basic.iloc[0].cost_estimate == pytest.approx(.033202)
