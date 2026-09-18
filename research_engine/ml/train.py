from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from research_engine.ml.baseline import make_model
from research_engine.ml.dataset import feature_columns
from research_engine.ml.importance import feature_importance
from research_engine.ml.walk_forward import day_splits


def _usable_features(frame: pd.DataFrame) -> List[str]:
    all_features = feature_columns(frame)
    return [name for name in all_features if not name.startswith(("call_", "put_")) or
            name.startswith(("call_atm_", "put_atm_", "call_buy_volume", "put_buy_volume",
                             "call_sell_volume", "put_sell_volume", "call_trade_imbalance",
                             "put_trade_imbalance", "call_volume", "put_volume",
                             "call_put_volume_ratio"))]


def walk_forward_train(dataset: pd.DataFrame, config: Dict, model_name: str = "logistic") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = config["walk_forward"]
    target = config["model"].get("classification_label", "future_direction_30s")
    regression = target.startswith("future_ret_")
    columns = _usable_features(dataset)
    days = sorted(dataset["date"].dropna().unique())
    predictions = []
    summaries = []
    importances = []
    for fold, split in enumerate(day_splits(
        days, int(settings["train_days"]), int(settings["validation_days"]),
        int(settings["test_days"]), bool(settings.get("expanding", True)),
    ), 1):
        train = dataset.loc[dataset["date"].isin(split.train) & dataset[target].notna()]
        validation = dataset.loc[dataset["date"].isin(split.validation) & dataset[target].notna()]
        test = dataset.loc[dataset["date"].isin(split.test) & dataset[target].notna()]
        if train.empty or test.empty or (not regression and train[target].nunique() < 2):
            continue
        # Validation days remain fully excluded from training, even if this
        # baseline does not tune a hyperparameter on them.
        model = make_model(model_name, "regression" if regression else "classification")
        x_train = train[columns].replace([np.inf, -np.inf], np.nan)
        x_test = test[columns].replace([np.inf, -np.inf], np.nan)
        # Entirely empty training columns are removed before the imputer.
        usable = x_train.columns[x_train.notna().any()].tolist()
        model.fit(x_train[usable], train[target])
        x_validation = validation[usable].replace([np.inf, -np.inf], np.nan)
        scored = test[list(dict.fromkeys(["timestamp", "date", target, "future_ret_30s", "mfe_30s", "mae_30s"]))].copy()
        scored["fold"] = fold
        if regression:
            scored["prediction"] = model.predict(x_test[usable])
            summaries.append({"fold": fold, "train_days": list(split.train), "validation_days": list(split.validation),
                              "test_days": list(split.test), "sample_count": len(test),
                              "validation_sample_count": len(validation),
                              "validation_mae": float(np.mean(np.abs(model.predict(x_validation) - validation[target]))) if len(validation) else None,
                              "mae": float(np.mean(np.abs(scored["prediction"] - test[target]))),
                              "correlation": scored["prediction"].corr(test[target])})
        else:
            probabilities = model.predict_proba(x_test[usable])
            classes = list(model.classes_)
            for cls in classes:
                scored["prob_" + str(cls).lower()] = probabilities[:, classes.index(cls)]
            predicted = model.predict(x_test[usable])
            scored["prediction"] = predicted
            true_up = (test[target] == "UP").astype(int).to_numpy()
            prob_up = scored.get("prob_up", pd.Series(np.zeros(len(test)), index=test.index)).to_numpy()
            summary = {
                "fold": fold, "train_days": list(split.train), "validation_days": list(split.validation),
                "test_days": list(split.test), "sample_count": len(test), "accuracy": accuracy_score(test[target], predicted),
                "validation_sample_count": len(validation),
                "precision_macro": precision_score(test[target], predicted, average="macro", zero_division=0),
                "recall_macro": recall_score(test[target], predicted, average="macro", zero_division=0),
                "f1_macro": f1_score(test[target], predicted, average="macro", zero_division=0),
                "confusion_labels": sorted(set(test[target]) | set(predicted)),
                "roc_auc_up": roc_auc_score(true_up, prob_up) if len(set(true_up)) > 1 else None,
                "pr_auc_up": average_precision_score(true_up, prob_up) if len(set(true_up)) > 1 else None,
            }
            labels_order = summary["confusion_labels"]
            summary["confusion_matrix"] = confusion_matrix(test[target], predicted, labels=labels_order).tolist()
            if len(validation):
                val_up = (validation[target] == "UP").astype(int).to_numpy()
                val_probs = model.predict_proba(x_validation)
                val_prob_up = val_probs[:, classes.index("UP")] if "UP" in classes else np.zeros(len(validation))
                summary["validation_roc_auc_up"] = roc_auc_score(val_up, val_prob_up) if len(set(val_up)) > 1 else None
                summary["validation_pr_auc_up"] = average_precision_score(val_up, val_prob_up) if len(set(val_up)) > 1 else None
            for threshold in config["model"]["probability_thresholds"]:
                selected = scored["prob_up"] > threshold if "prob_up" in scored else pd.Series(False, index=scored.index)
                subset = scored.loc[selected]
                prefix = "prob_up_gt_{}_".format(threshold)
                summary.update({
                    prefix + "count": int(selected.sum()),
                    prefix + "precision": float((subset[target] == "UP").mean()) if len(subset) else None,
                    prefix + "future_return": float(subset["future_ret_30s"].mean()) if len(subset) else None,
                    prefix + "mfe": float(subset["mfe_30s"].mean()) if len(subset) else None,
                    prefix + "mae": float(subset["mae_30s"].mean()) if len(subset) else None,
                })
            summaries.append(summary)
        importance = feature_importance(model, usable)
        importance["fold"] = fold
        importances.append(importance)
        predictions.append(scored)
    if not predictions:
        raise ValueError("No walk-forward folds: need at least {} separate dates, found {}".format(
            settings["train_days"] + settings["validation_days"] + settings["test_days"], len(days)))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries), pd.concat(importances, ignore_index=True)
