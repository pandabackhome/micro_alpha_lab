import pandas as pd

from research_engine.analysis.correlations import correlations
from research_engine.analysis.quantiles import quantile_analysis
from research_engine.analysis.regimes import evaluate_rules


def test_quantile_preserves_ties_instead_of_fake_time_order_signal():
    frame = pd.DataFrame({"ofi_5s": [0, 0, 0, 0, 1, 1, 1, 1],
                          "future_ret_30s": [-1, -1, -1, -1, 1, 1, 1, 1]})
    result = quantile_analysis(frame, "ofi_5s", ["future_ret_30s"], bins=10)
    assert result["sample_count"].sum() == 8
    assert len(result) <= 2


def test_configurable_rules_and_per_day_ic():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-14T13:30:00Z", "2026-09-14T13:30:01Z",
                                     "2026-09-14T13:30:02Z", "2026-09-15T13:30:00Z",
                                     "2026-09-15T13:30:01Z", "2026-09-15T13:30:02Z"]),
        "depth_imbalance": [0.7, 0.2, 0.3, 0.8, 0.1, 0.4],
        "trade_imbalance_5s": [0.6, 0.6, 0.1, 0.3, 0.2, 0.2],
        "future_ret_30s": [0.01, -0.01, 0.001, 0.02, -0.02, 0.002],
        "mfe_30s": [0.02, 0, 0.001, 0.03, 0, 0.002],
        "mae_30s": [-0.01, -0.02, -0.01, -0.01, -0.03, -0.002],
    })
    rules = evaluate_rules(frame, [{"name": "buy", "conditions": {
        "depth_imbalance": "> 0.6", "trade_imbalance_5s": "> 0.5"}}])
    assert rules.loc[0, "signal_count"] == 1
    assert rules.loc[0, "future_return"] == 0.01
    result = correlations(frame, ["depth_imbalance"], ["future_ret_30s"])
    assert set(result["scope"]) == {"overall", "2026-09-14", "2026-09-15"}
