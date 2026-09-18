from __future__ import annotations

import numpy as np
import pandas as pd


def feature_importance(model, names):
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "feature_importances_"):
        if not hasattr(estimator, "coef_"):
            return pd.DataFrame(columns=["feature", "importance"])
        values = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        values = estimator.feature_importances_
    labels = list(names) + ["missing_indicator_{}".format(i) for i in range(len(values) - len(names))]
    return pd.DataFrame({"feature": labels[:len(values)], "importance": values}).sort_values("importance", ascending=False)

