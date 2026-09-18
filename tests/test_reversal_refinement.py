import pandas as pd
import pytest

from research_engine.analysis.reversal_refinement import efficiency_thresholds, refine


T = pd.Timestamp('2026-09-15T14:00:30Z')


def source():
    return pd.DataFrame(dict(timestamp=[T], signal_status='signal', direction=1, spot=100., spread=.02,
                             efficiency=.1, efficiency_median=.2, factor='reversal60'))


def quotes():
    return pd.DataFrame(dict(timestamp=pd.date_range(T-pd.Timedelta(seconds=5), periods=46, freq='1s'),
                             qqq_bid=99.99, qqq_ask=100.01, qqq_depth_age_seconds=0.))


def test_confirmation_uses_first_observed_turn_not_best_future_price():
    frame = quotes()
    frame.loc[frame.timestamp.eq(T+pd.Timedelta(seconds=3)), ['qqq_bid','qqq_ask']] += .03
    result = refine(source(), frame, 'turn_confirmed').iloc[0]
    assert result.timestamp == T + pd.Timedelta(seconds=3)
    assert result.spot == pytest.approx(100.03)
    frame.loc[frame.timestamp.gt(T+pd.Timedelta(seconds=3)), ['qqq_bid','qqq_ask']] += 50
    changed = refine(source(), frame, 'turn_confirmed').iloc[0]
    assert result.equals(changed)
    assert result.origin_timestamp == T


def test_no_turn_times_out_and_high_efficiency_is_filtered_before_confirmation():
    result = refine(source(), quotes(), 'turn_confirmed').iloc[0]
    assert result.signal_status == 'confirmation_timeout' and result.direction == 0
    signal = source()
    signal['efficiency'] = .3
    result = refine(signal, quotes(), 'range_and_turn').iloc[0]
    assert result.signal_status == 'range_filtered'
    assert result.timestamp == T


def test_efficiency_thresholds_require_prior_dates():
    current = pd.DataFrame(dict(timestamp=[T], date='2026-09-15', half_hour=1, efficiency=.3, valid300=True))
    history = []
    for date in ['2026-08-31','2026-09-01','2026-09-02','2026-09-03','2026-09-04']:
        history.append(pd.DataFrame(dict(date=date, half_hour=[1]*25, efficiency=.2, valid300=True)))
    assert efficiency_thresholds(current, history).iloc[0].efficiency_median == .2
    history[-1]['date'] = '2026-09-15'
    with pytest.raises(ValueError, match='earlier'):
        efficiency_thresholds(current, history)
