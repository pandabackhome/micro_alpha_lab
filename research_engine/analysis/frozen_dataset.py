"""Resolve exact research inputs despite concurrent archive additions."""
from __future__ import annotations

import json
from pathlib import Path

from research_engine.analysis.causal_volatility import file_hash


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "research/stock_search_dataset.json"


def research_inputs(archive_data, local_data, manifest_path=DEFAULT_MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text())
    roots = {"archive": Path(archive_data), "local": Path(local_data)}
    records = {}
    for item in manifest["sources"]:
        path = roots[item["origin"]] / item["file"]
        if file_hash(path) != item["sha256"]:
            raise ValueError("frozen source changed: " + item["file"])
        records[(item["origin"],item["file"])] = path
    result = {}
    for day in manifest["original_dates"] + manifest["added_dates"]:
        origin = "archive" if day in manifest["original_dates"] else "local"
        feature = records[(origin,"features/"+day+".parquet")]
        normalized = records[(origin,"normalized/"+day+"/events.parquet")]
        result[day] = (origin, feature, normalized)
    if len(result) != len(manifest["original_dates"]) + len(manifest["added_dates"]):
        raise ValueError("duplicate date in frozen manifest")
    return result, manifest
