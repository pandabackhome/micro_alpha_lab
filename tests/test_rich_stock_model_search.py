import numpy as np
import pandas as pd
import pytest

from research_engine.analysis.rich_stock_model_search import FEATURES, fitted_predictions


def test_rich_pipeline_imputes_from_past_and_ignores_current_targets():
    rng=np.random.RandomState(42)
    history=[]
    for date in ['2026-09-04','2026-09-08','2026-09-09','2026-09-10','2026-09-11']:
        frame=pd.DataFrame(rng.normal(size=(40,len(FEATURES))),columns=FEATURES)
        frame['date'],frame['valid300']=date,True
        frame['target_CALL_300'],frame['target_PUT_300']=3.,-2.
        frame.loc[:5,'ofi_unrelated_future_label']=10000.
        frame.loc[:5,'microprice_delta_bps']=np.nan
        history.append(frame)
    current=history[-1].iloc[:3].copy().assign(date='2026-09-14')
    before,audit=fitted_predictions(current,history,300,'rich_ridge',min_samples=20)
    current['target_CALL_300'],current['target_PUT_300']=-1e9,1e9
    after,_=fitted_predictions(current,history,300,'rich_ridge',min_samples=20)
    np.testing.assert_allclose(before['CALL'],3.)
    np.testing.assert_allclose(before['PUT'],-2.)
    np.testing.assert_array_equal(before['CALL'],after['CALL'])
    assert all(row['history_last_date']=='2026-09-11' for row in audit)
    with pytest.raises(ValueError,match='earlier'):
        fitted_predictions(current,history[:4]+[current],300,'rich_ridge',min_samples=20)
