from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from research_engine.analysis.correlations import correlations
from research_engine.analysis.feature_stats import feature_statistics
from research_engine.analysis.quantiles import quantile_analysis
from research_engine.analysis.regimes import evaluate_rules
from research_engine.backtest.metrics import backtest_metrics
from research_engine.ml.dataset import feature_columns


DEFAULT_FEATURES = [
    "depth_imbalance", "trade_imbalance_5s", "ofi_5s", "microprice_delta_bps",
    "call_atm_option_volume_burst", "put_atm_option_volume_burst",
]


def generate_report(dataset: pd.DataFrame, config: Dict, output_root: Path,
                    predictions: Optional[pd.DataFrame] = None,
                    folds: Optional[pd.DataFrame] = None,
                    importance: Optional[pd.DataFrame] = None,
                    trades: Optional[pd.DataFrame] = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = output_root / stamp
    if output.exists():
        output = output_root / (stamp + "_{}".format(datetime.now(timezone.utc).microsecond))
    output.mkdir(parents=True)
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    clean_config["paths"] = dict(clean_config["paths"])
    home = Path.home()
    for key, value in clean_config["paths"].items():
        path = Path(value)
        if path.is_absolute():
            try:
                clean_config["paths"][key] = "~/" + str(path.relative_to(home))
            except ValueError:
                pass
    (output / "config.yaml").write_text(yaml.safe_dump(clean_config, sort_keys=False), encoding="utf-8")
    feature_statistics(dataset[feature_columns(dataset)]).to_csv(output / "feature_stats.csv", index=False)
    usable = [name for name in DEFAULT_FEATURES if name in dataset]
    quantiles = pd.concat([quantile_analysis(dataset, name) for name in usable], ignore_index=True) if usable else pd.DataFrame()
    quantiles.to_csv(output / "quantile_analysis.csv", index=False)
    correlation = correlations(dataset, usable, [name for name in ("future_ret_10s", "future_ret_30s", "future_ret_60s") if name in dataset])
    correlation.to_csv(output / "correlation.csv", index=False)
    rules = evaluate_rules(dataset, clean_config.get("rules", []))
    rules.to_csv(output / "rule_analysis.csv", index=False)
    summary = {
        "dataset_rows": len(dataset), "days": sorted(dataset["date"].unique().tolist()),
        "quantile_features": usable, "walk_forward_folds": len(folds) if folds is not None else 0,
    }
    if predictions is not None:
        predictions.to_parquet(output / "walk_forward_predictions.parquet", index=False)
    if folds is not None:
        folds.to_json(output / "walk_forward_folds.json", orient="records", indent=2)
        summary["walk_forward"] = folds.to_dict(orient="records")
    if importance is not None:
        importance.to_csv(output / "feature_importance.csv", index=False)
    if trades is not None:
        trades.to_csv(output / "backtest_trades.csv", index=False)
        pd.DataFrame(trades.attrs.get("unclosed_records", [])).to_csv(output / "unclosed_positions.csv", index=False)
        metrics, daily = backtest_metrics(trades)
        daily.to_csv(output / "daily_results.csv", index=False)
        summary["backtest"] = metrics
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# QQQ Microstructure Research Report", "",
        "Generated: {}".format(datetime.now(timezone.utc).isoformat()), "",
        "Days: {}. Sampled snapshots: {}.".format(len(summary["days"]), len(dataset)), "",
        "No raw recording was modified. UTC availability is max(event_ts, received_at); labels are separate Parquet.", "",
        "## Walk-forward", "",
    ]
    if folds is not None and not folds.empty:
        for fold in folds.to_dict(orient="records"):
            lines.append("- Fold {}: train {} days, validate {}, test {}; accuracy {}, UP ROC-AUC {}, UP PR-AUC {}.".format(
                fold["fold"], len(fold["train_days"]), ", ".join(fold["validation_days"]),
                ", ".join(fold["test_days"]), fold.get("accuracy"), fold.get("roc_auc_up"), fold.get("pr_auc_up")))
    else:
        lines.append("No walk-forward result generated.")
    lines += ["", "## Backtest", ""]
    if trades is not None:
        lines.append("Mode: {}. Trades: {}. Total PnL: {}. Win rate: {}.".format(
            config["execution"]["mode"], summary["backtest"]["trade_count"],
            summary["backtest"].get("total_pnl"), summary["backtest"].get("win_rate")))
        lines.append("Execution uses next snapshot ask/bid and configured slippage/commission; no fill guarantee or market impact modeled.")
    else:
        lines.append("No backtest run.")
    lines += ["", "## Research artifacts", "", "Quantile, correlation, rule, feature-statistics and daily results are saved beside this report.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output
