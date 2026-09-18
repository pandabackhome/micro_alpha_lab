import pandas as pd
import pytest

from research_engine.analysis.event_reversal_search import crossing_signals


def test_first_crossing_is_not_repeated_inside_extreme_region_or_on_first_valid_row():
    frame = pd.DataFrame(dict(timestamp=pd.date_range('2026-09-15T14:00:00Z', periods=9, freq='1s'),
                              ready=True, valid300=True, seconds_from_open=1800,
                              threshold_r60=.01, r60=[.02,.02,.001,.02,.02,.001,-.02,-.02,.02]))
    result = crossing_signals(frame)
    assert result.scheduled.tolist() == [False,False,False,True,False,False,True,False,True]
    assert result.direction.tolist() == [0,0,0,-1,0,0,1,0,-1]
    frame.loc[7:,'r60'] = .5
    later = crossing_signals(frame)
    pd.testing.assert_frame_equal(result.iloc[:7], later.iloc[:7])
    with pytest.raises(ValueError, match='one-second'):
        crossing_signals(frame.drop(index=4))


def test_recovery_requires_leaving_extreme_and_turn_requires_contemporaneous_price_turn():
    frame = pd.DataFrame(dict(timestamp=pd.date_range('2026-09-15T14:00:00Z', periods=7, freq='1s'),
                              ready=True, valid300=True, seconds_from_open=1800,
                              threshold_r60=.01, r60=[.02,.015,.005,.02,.005,-.02,-.005],
                              r5=[.001,.001,-.001,.001,.001,-.001,.001]))
    raw = crossing_signals(frame, mode='recovery')
    turn = crossing_signals(frame, mode='recovery_turn')
    assert raw[raw.scheduled].index.tolist() == [2,4,6]
    assert turn[turn.scheduled].index.tolist() == [2,6]
    assert turn[turn.scheduled].direction.tolist() == [-1,1]
