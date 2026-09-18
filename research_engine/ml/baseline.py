from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str = "logistic", task: str = "classification"):
    if name == "logistic":
        # Heavily correlated rolling features and an imbalanced FLAT class
        # benefit from stronger regularization; scale before LBFGS.
        estimator = LogisticRegression(max_iter=1500, tol=1e-3, C=0.03, class_weight="balanced") if task == "classification" else Ridge()
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), estimator)
    if name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=30, n_jobs=-1, random_state=42) if task == "classification" else RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=30, n_jobs=-1, random_state=42)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)
    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install '.[lightgbm]'") from exc
        estimator = lgb.LGBMClassifier(n_estimators=150, max_depth=6, random_state=42, verbosity=-1) if task == "classification" else lgb.LGBMRegressor(n_estimators=150, max_depth=6, random_state=42, verbosity=-1)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install '.[xgboost]'") from exc
        estimator = xgb.XGBClassifier(n_estimators=150, max_depth=6, random_state=42) if task == "classification" else xgb.XGBRegressor(n_estimators=150, max_depth=6, random_state=42)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)
    if name == "lstm":
        # Not wrapped in a pipeline: the imputer and scaler have to run before
        # the rows are cut into windows, so the estimator owns them. It also
        # needs the session key per row, which a pipeline has no way to pass.
        from research_engine.ml.sequence import SequenceClassifier
        return SequenceClassifier(task=task)
    raise ValueError("unknown model: {}".format(name))
