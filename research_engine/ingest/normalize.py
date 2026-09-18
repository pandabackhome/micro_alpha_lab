from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

from research_engine.config import config_hash, portable_path
from research_engine.ingest.option_symbol import parse_option_symbol
from research_engine.ingest.parser import event_times
from research_engine.ingest.reader import iter_events, read_metadata


SCHEMA = pa.schema(
    [
        ("line_number", pa.int64()),
        ("kind", pa.string()),
        ("symbol", pa.string()),
        ("event_ts", pa.timestamp("us", tz="UTC")),
        ("received_at", pa.timestamp("us", tz="UTC")),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("event_timestamp_source", pa.string()),
        ("price", pa.float64()),
        ("volume", pa.float64()),
        ("direction", pa.string()),
        ("trade_type", pa.string()),
        ("trade_session", pa.string()),
        ("bid_prices", pa.list_(pa.float64())),
        ("bid_sizes", pa.list_(pa.float64())),
        ("ask_prices", pa.list_(pa.float64())),
        ("ask_sizes", pa.list_(pa.float64())),
        ("best_bid", pa.float64()),
        ("best_bid_size", pa.float64()),
        ("best_ask", pa.float64()),
        ("best_ask_size", pa.float64()),
        ("option_expiry", pa.date32()),
        ("option_right", pa.string()),
        ("option_strike", pa.float64()),
    ]
)


def source_hash(path: Union[str, Path]) -> str:
    path = Path(path)
    sidecar = path.with_name(path.name.replace(".jsonl.gz", ".jsonl.source.sha256"))
    if sidecar.exists():
        token = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if len(token) == 64:
            return token
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "compressed:" + digest.hexdigest()


def _levels(values: Any) -> Tuple[List[float], List[float]]:
    levels = []
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            if item[1] is None or item[2] is None:
                continue
            price, size = float(item[1]), float(item[2])
            if price > 0 and size >= 0:
                levels.append((int(item[0]), price, size))
    levels.sort(key=lambda x: x[0])
    return [x[1] for x in levels], [x[2] for x in levels]


def normalize_payload(line_number: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    event_ts, received, available, timestamp_source = event_times(payload)
    bid_prices, bid_sizes = _levels(payload.get("bids"))
    ask_prices, ask_sizes = _levels(payload.get("asks"))
    contract = parse_option_symbol(str(payload.get("symbol", "")))
    return {
        "line_number": line_number,
        "kind": str(payload.get("kind", "unknown")),
        "symbol": payload.get("symbol"),
        "event_ts": event_ts,
        "received_at": received,
        "available_at": available,
        "event_timestamp_source": timestamp_source,
        "price": _number(payload.get("price")),
        "volume": _number(payload.get("volume")),
        "direction": payload.get("direction"),
        "trade_type": payload.get("trade_type"),
        "trade_session": payload.get("trade_session"),
        "bid_prices": bid_prices or None,
        "bid_sizes": bid_sizes or None,
        "ask_prices": ask_prices or None,
        "ask_sizes": ask_sizes or None,
        "best_bid": bid_prices[0] if bid_prices else None,
        "best_bid_size": bid_sizes[0] if bid_sizes else None,
        "best_ask": ask_prices[0] if ask_prices else None,
        "best_ask_size": ask_sizes[0] if ask_sizes else None,
        "option_expiry": contract.expiry if contract else None,
        "option_right": contract.call_put if contract else None,
        "option_strike": contract.strike if contract else None,
    }


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def _cache_valid(output: Path, expected: Dict[str, Any]) -> bool:
    meta = output.with_suffix(output.suffix + ".meta.json")
    if not output.exists() or not meta.exists():
        return False
    try:
        current = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all(current.get(key) == value for key, value in expected.items())


def normalize_recording(
    source: Union[str, Path],
    output: Union[str, Path],
    config: Dict[str, Any],
    *,
    force: bool = False,
    batch_size: int = 100_000,
) -> Dict[str, Any]:
    source, output = Path(source), Path(output)
    expected = {
        "source": portable_path(source),
        "source_hash": source_hash(source),
        "config_hash": config_hash(config, "normalized"),
        "dedup_mode": config.get("dedup_mode", "none"),
        "schema_version": 1,
    }
    if not force and _cache_valid(output, expected):
        result = dict(expected)
        result.update({"output": str(output), "cached": True})
        return result

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(output), SCHEMA, compression="zstd")
    rows: List[Dict[str, Any]] = []
    count = 0
    try:
        for wrapped in iter_events(source, dedup_mode=config.get("dedup_mode", "none"), include_metadata=False):
            rows.append(normalize_payload(wrapped.line_number, wrapped.payload))
            if len(rows) >= batch_size:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                count += len(rows)
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            count += len(rows)
    finally:
        writer.close()
    metadata = read_metadata(source)
    result = dict(expected)
    result.update({"output": str(output), "cached": False, "row_count": count, "recording_metadata": metadata})
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(dict(result, output=portable_path(output)), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result


def merge_normalized(sources: Sequence[Union[str, Path]], output: Union[str, Path], config: Dict,
                     *, force: bool = False) -> Dict[str, Any]:
    """Append multiple independent captures of one session, preserving every event.

    The captures remain distinguishable by their individual Parquet files.
    Neither capture is discarded or deduplicated; downstream sorts availability.
    """
    sources, output = [Path(x) for x in sources], Path(output)
    if len(sources) == 1:
        return normalize_recording(sources[0], output, config, force=force)
    parts = []
    for source in sources:
        part = output.parent / (source.name.replace(".jsonl.gz", "") + ".parquet")
        parts.append(normalize_recording(source, part, config, force=force))
    digest = hashlib.sha256("|".join(x["source_hash"] for x in parts).encode()).hexdigest()
    expected = {"source_hash": digest, "config_hash": config_hash(config, "normalized"), "schema_version": 1,
                "sources": [x["source"] for x in parts], "source_hashes": [x["source_hash"] for x in parts]}
    if not force and _cache_valid(output, expected):
        return dict(expected, output=str(output), cached=True)
    writer = pq.ParquetWriter(str(output), SCHEMA, compression="zstd")
    count = 0
    try:
        for part in parts:
            file = pq.ParquetFile(part["output"])
            for batch in file.iter_batches(batch_size=100_000):
                writer.write_batch(batch)
                count += batch.num_rows
    finally:
        writer.close()
    result = dict(expected, output=str(output), cached=False, row_count=count)
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(dict(result, output=portable_path(output)), indent=2) + "\n", encoding="utf-8"
    )
    return result
