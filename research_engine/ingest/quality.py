from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import chain
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple, Union

from research_engine.ingest.option_symbol import parse_option_symbol
from research_engine.ingest.parser import best_level, event_times
from research_engine.ingest.reader import iter_events
from research_engine.models import QualityReport


def inspect_recording(path: Union[str, Path, List[Union[str, Path]]], underlying: str = "QQQ.US") -> QualityReport:
    """Inspect one or several captures of the same date without dropping prints."""
    paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
    report = QualityReport(source=", ".join(str(p) for p in paths))
    counts: Counter = Counter()
    symbols = set()
    strikes = set()
    seen_trade_fingerprints = set()
    previous_event = None
    previous_received = None
    comparable_event = comparable_received = 0
    event_lags: List[float] = []
    min_event = max_event = min_received = max_received = None

    try:
        iterator = chain.from_iterable(iter_events(source) for source in paths)
        for wrapped in iterator:
            payload = wrapped.payload
            kind = wrapped.kind
            counts[kind] += 1
            report.event_count += 1
            symbol = payload.get("symbol")
            if symbol:
                symbols.add(symbol)
                contract = parse_option_symbol(symbol)
                if symbol == underlying:
                    report.qqq_event_count += 1
                elif contract:
                    report.option_event_count += 1
                    strikes.add(contract.strike)

            event_ts, received, _, source = event_times(payload)
            if source == "received_fallback":
                report.depth_event_timestamp_fallback += 1
            if kind not in {"header", "contracts"} and event_ts is None:
                report.missing_timestamp += 1
            if event_ts:
                min_event = event_ts if min_event is None or event_ts < min_event else min_event
                max_event = event_ts if max_event is None or event_ts > max_event else max_event
                if previous_event is not None:
                    comparable_event += 1
                    if event_ts < previous_event:
                        report.out_of_order_event_count += 1
                previous_event = event_ts
            if received:
                min_received = received if min_received is None or received < min_received else min_received
                max_received = received if max_received is None or received > max_received else max_received
                if previous_received is not None:
                    comparable_received += 1
                    if received < previous_received:
                        report.out_of_order_received_count += 1
                previous_received = received
            if event_ts and received:
                event_lags.append((received - event_ts).total_seconds() * 1000.0)

            price = payload.get("price")
            if kind in {"trade", "quote", "bar"} and (
                not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0
            ):
                report.invalid_price += 1
            if kind == "depth":
                bid, _ = best_level(payload.get("bids"))
                ask, _ = best_level(payload.get("asks"))
                if bid is not None and ask is not None and bid > ask:
                    report.invalid_spread += 1
            if kind == "trade":
                fields = (
                    symbol,
                    payload.get("ts"),
                    payload.get("received_at"),
                    payload.get("price"),
                    payload.get("volume"),
                    payload.get("direction"),
                    payload.get("trade_type"),
                    payload.get("trade_session"),
                )
                digest = hashlib.blake2b(repr(fields).encode("utf-8"), digest_size=8).digest()
                if digest in seen_trade_fingerprints:
                    report.suspected_duplicate_count += 1
                else:
                    seen_trade_fingerprints.add(digest)
    except json.JSONDecodeError:
        report.malformed_json += 1
        raise

    report.counts_by_kind = dict(sorted(counts.items()))
    trade_count = counts.get("trade", 0)
    report.suspected_duplicate_ratio = report.suspected_duplicate_count / trade_count if trade_count else 0.0
    report.out_of_order_ratio = report.out_of_order_event_count / comparable_event if comparable_event else 0.0
    report.out_of_order_received_ratio = (
        report.out_of_order_received_count / comparable_received if comparable_received else 0.0
    )
    report.first_event_timestamp = min_event.isoformat() if min_event else None
    report.last_event_timestamp = max_event.isoformat() if max_event else None
    report.first_received_timestamp = min_received.isoformat() if min_received else None
    report.last_received_timestamp = max_received.isoformat() if max_received else None
    report.symbols = sorted(symbols)
    report.symbol_count = len(symbols)
    report.option_strikes = sorted(strikes)
    report.option_strike_count = len(strikes)
    if event_lags:
        ordered = sorted(event_lags)
        report.event_lag_ms = {
            "min": ordered[0],
            "median": median(ordered),
            "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
            "max": ordered[-1],
        }
    if report.depth_event_timestamp_fallback:
        report.notes.append("Depth records have no exchange event timestamp; received_at is used as event_ts.")
    if report.suspected_duplicate_count:
        report.notes.append("Suspected duplicate trades are reported but retained; no exchange id/sequence is available.")
    if report.out_of_order_received_count:
        report.notes.append("Source lines are not strictly ordered by received_at; downstream uses available_at sorting/as-of logic.")
    return report
