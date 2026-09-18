import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import StudySettings
from research_engine.analysis.long_context_search import long_observations, long_signals, recorded_vwap, thresholds


def test_recorded_vwap_uses_availability_and_never_backfills_future_prints():
    t=pd.Timestamp('2026-09-17T13:30:00Z')
    grid=pd.date_range(t,periods=5,freq='1s')
    trades=pd.DataFrame(dict(available_at=[t-pd.Timedelta(seconds=1),t+pd.Timedelta(seconds=1),t+pd.Timedelta(seconds=3)],
                             price=[1000.,100.,110.],volume=[100.,1.,3.]))
    values=recorded_vwap(grid,trades,t)
    np.testing.assert_allclose(values,[np.nan,100.,100.,107.5,107.5],equal_nan=True)
    trades.loc[2,'price']=10000.
    np.testing.assert_array_equal(values[:3],recorded_vwap(grid,trades,t)[:3])


def test_long_features_and_historical_thresholds_do_not_see_future_data():
    t=pd.Timestamp('2026-09-17T13:30:00Z')
    grid=pd.date_range(t,periods=3601,freq='1s')
    prices=100+np.sin(np.arange(len(grid))/30.)*.1
    frame=pd.DataFrame(dict(timestamp=grid,qqq_bid=prices-.01,qqq_ask=prices+.01,qqq_depth_age_seconds=0.,
                            bid_size=100.,ask_size=100.,buy_volume_30s=100.,sell_volume_30s=50.,total_volume_5s=30.))
    trades=pd.DataFrame(dict(available_at=grid[::10],price=prices[::10],volume=100.))
    initial=long_observations(frame,trades,StudySettings())
    frame.loc[2000:,['qqq_bid','qqq_ask']]+=10.
    trades.loc[trades.available_at.ge(grid[2000]),'price']+=10.
    later=long_observations(frame,trades,StudySettings())
    pd.testing.assert_frame_equal(initial[initial.timestamp.lt(grid[2000])],later[later.timestamp.lt(grid[2000])])
    history=[initial[initial.seconds_from_open.mod(30).eq(0)].assign(date=date)
             for date in ['2026-09-04','2026-09-08','2026-09-11','2026-09-14','2026-09-15']]
    before=thresholds(initial,history)
    after=thresholds(later,history)
    pd.testing.assert_series_equal(before.threshold_r900,after.threshold_r900)
    assert before.long_ready.any()
    with pytest.raises(ValueError,match='earlier'):
        thresholds(initial,history[:4]+[initial])


def test_opening_retest_waits_after_initial_breakout():
    frame=pd.DataFrame(dict(spot=[100.,101.,100.,101.],previous_spot=[99.,100.,101.,100.],r900=.01,
                            vwap_deviation=.01,previous_vwap_deviation=.01,threshold_r900=.001,
                            volume5=10.,threshold_volume5=20.,r60=.001,r5=.001,threshold_vwap_deviation=.02,
                            high300=110.,low300=90.,opening_high=100.,opening_low=90.,
                            seconds_from_open=[900,901,960,961],first_opening_up=[np.nan,901,901,901],
                            first_opening_down=np.nan,long_valid=True,long_ready=True))
    result=long_signals(frame)
    result=result[result.factor.eq('opening_retest')]
    assert result.direction.tolist()==[0,0,0,1]
