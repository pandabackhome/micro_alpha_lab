from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from research_engine.analysis.quantiles import quantile_analysis
from research_engine.backtest.engine import backtest_day
from research_engine.backtest.metrics import backtest_metrics
from research_engine.backtest.strategies import ml_probability
from research_engine.config import load_config, resolve_path
from research_engine.features.pipeline import write_feature_outputs
from research_engine.ingest.normalize import merge_normalized
from research_engine.ingest.quality import inspect_recording
from research_engine.ingest.reader import read_metadata
from research_engine.labels.forward_returns import write_labels
from research_engine.ml.dataset import load_dataset
from research_engine.ml.train import walk_forward_train
from research_engine.reports.generate import generate_report


def recording_index(config, verbose=True):
    """Index by actual contracts expiry, not filename (one file is misdated)."""
    candidates = defaultdict(list)
    for source in sorted(resolve_path(config, "recordings").glob("*.jsonl.gz")):
        metadata = read_metadata(source)
        contracts = metadata.get("contracts", {})
        day = contracts.get("expiry")
        if not day or contracts.get("underlying") != config["underlying"]:
            continue
        # Out-of-session short diagnostic captures and duplicate same-day
        # captures are not silently merged. Prefer the main daily capture;
        # inspect can diagnose secondary sources independently.
        started = metadata.get("header", {}).get("started_at", "")
        if started[:10] != day or started[11:13] < "12":
            if verbose:
                print("WARNING {}: excluded out-of-session/previous-day capture {} (started {}).".format(
                    day, source.name, started), file=sys.stderr)
            continue
        candidates[day].append(source)
    chosen = {}
    for day, files in candidates.items():
        chosen[day] = sorted(files)
        if len(files) > 1 and verbose:
            print("INFO {}: merging {} main-session captures without dedup: {}.".format(
                day, len(files), ", ".join(p.name for p in chosen[day])), file=sys.stderr)
    return chosen


def build_day(config, day: str, force: bool = False):
    index = recording_index(config, verbose=False)
    if day not in index:
        raise ValueError("no {} recording found for {}".format(config["underlying"], day))
    root = resolve_path(config, "data")
    normalized = root / "normalized" / day / "events.parquet"
    snapshot = root / "snapshots" / (day + ".parquet")
    features = root / "features" / (day + ".parquet")
    labels = root / "labels" / (day + ".parquet")
    n = merge_normalized(index[day], normalized, config, force=force)
    f = write_feature_outputs(normalized, date.fromisoformat(day), snapshot, features, config, force=force)
    # Labels are quick and always rebuilt so changed label horizons/thresholds
    # cannot be mistaken for an earlier configuration's results.
    l = write_labels(features, labels, config)
    print(json.dumps({"date": day, "sources": [p.name for p in index[day]], "normalize_cached": n["cached"],
                      "feature_cached": f["cached"], "rows": l["rows"], "feature_columns": f.get("columns")},
                     default=str), flush=True)


def _paths(config):
    return sorted((resolve_path(config, "data") / "features").glob("*.parquet"))


def _backtest(config, predictions, threshold):
    signals = ml_probability(predictions, threshold)
    trades = []
    skipped_entries = skipped_late_entries = unclosed_positions = 0
    unclosed_records = []
    for day, group in predictions.groupby("date"):
        feature_path = resolve_path(config, "data") / "features" / (day + ".parquet")
        if feature_path.exists():
            frame = pd.read_parquet(feature_path)
            subset = signals[signals["timestamp"].isin(group["timestamp"])]
            result = backtest_day(frame, subset, config)
            skipped_entries += result.attrs.get("skipped_entries", 0)
            skipped_late_entries += result.attrs.get("skipped_late_entries", 0)
            unclosed_positions += result.attrs.get("unclosed_positions", 0)
            unclosed_records.extend(result.attrs.get("unclosed_records", []))
            if not result.empty:
                trades.append(result)
    result = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    result.attrs["skipped_entries"] = skipped_entries
    result.attrs["skipped_late_entries"] = skipped_late_entries
    result.attrs["unclosed_positions"] = unclosed_positions
    result.attrs["unclosed_records"] = unclosed_records
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="QQQ/0DTE leakage-aware microstructure research")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--recordings-root", type=Path,
                        help="read raw recordings from this directory; derived data stays in the configured data directory")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="stream a recording and report data quality")
    inspect_source = inspect.add_mutually_exclusive_group(required=True)
    inspect_source.add_argument("--input", type=Path)
    inspect_source.add_argument("--date", help="aggregate all main-session captures of YYYY-MM-DD")
    inspect.add_argument("--output", type=Path, help="also save the JSON report")
    build = sub.add_parser("build-features", help="normalize, snapshot, features, labels")
    build.add_argument("--date")
    build.add_argument("--all", action="store_true")
    build.add_argument("--force", action="store_true")
    analyze = sub.add_parser("analyze", help="real quantile analysis over built dates")
    analyze.add_argument("--feature", required=True)
    analyze.add_argument("--label", default="future_ret_30s")
    analyze.add_argument("--stride", type=int, default=5)
    analyze.add_argument("--data-root", type=Path, help="read existing features and labels from this directory")
    train = sub.add_parser("train", help="day-split baseline model")
    train.add_argument("--model", default="logistic", choices=["logistic", "random_forest", "lightgbm", "xgboost", "lstm"])
    train.add_argument("--stride", type=int, default=10)
    train.add_argument("--data-root", type=Path, help="read existing features and labels from this directory")
    backtest = sub.add_parser("backtest", help="next-available ask/bid simulation")
    backtest.add_argument("--strategy", default="ml_probability", choices=["ml_probability"])
    backtest.add_argument("--model", default="logistic")
    backtest.add_argument("--mode", default="underlying", choices=["underlying", "option"])
    backtest.add_argument("--threshold", type=float, default=0.75)
    backtest.add_argument("--stride", type=int, default=10)
    backtest.add_argument("--data-root", type=Path, help="read existing features and labels from this directory")
    all_parser = sub.add_parser("run-all", help="build all dates, analyze, train, backtest, report")
    all_parser.add_argument("--model", default="logistic")
    all_parser.add_argument("--stride", type=int, default=10)
    all_parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.recordings_root is not None:
        config["paths"]["recordings"] = str(args.recordings_root.resolve())
    if getattr(args, "data_root", None) is not None:
        config["paths"]["data"] = str(args.data_root.resolve())
    if args.command == "inspect":
        if args.date:
            sources = recording_index(config).get(args.date)
            if not sources:
                parser.error("no main-session QQQ capture found for {}".format(args.date))
        else:
            sources = args.input
        payload = json.dumps(inspect_recording(sources, config["underlying"]).to_dict(), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return
    if args.command in {"build-features", "run-all"}:
        if args.command == "build-features" and not args.all and not args.date:
            parser.error("build-features requires --date YYYY-MM-DD or --all")
        days = [args.date] if args.command == "build-features" and args.date else sorted(recording_index(config))
        for day in days:
            build_day(config, day, force=getattr(args, "force", False))
        if args.command == "build-features":
            return
    paths = _paths(config)
    if not paths:
        parser.error("no built features found; run build-features first")
    if args.command == "analyze":
        dataset = load_dataset(paths, label=args.label, stride=args.stride)
        print(quantile_analysis(dataset, args.feature, [args.label]).to_string(index=False))
        return
    dataset = load_dataset(paths, label=config["model"]["classification_label"], stride=args.stride)
    predictions, folds, importance = walk_forward_train(dataset, config, model_name=args.model)
    if args.command == "train":
        print(folds.to_json(orient="records", indent=2))
        return
    if args.command == "backtest":
        config["execution"]["mode"] = args.mode
    trades = _backtest(config, predictions, args.threshold)
    output = generate_report(dataset, config, resolve_path(config, "results"), predictions, folds, importance, trades)
    print(json.dumps({"report": str(output), "folds": folds.to_dict(orient="records"),
                      "backtest": backtest_metrics(trades)[0]}, indent=2, default=str))


if __name__ == "__main__":
    main()
