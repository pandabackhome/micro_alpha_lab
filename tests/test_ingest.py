import gzip
import json
from datetime import date

import pytest

from research_engine.ingest.option_symbol import parse_option_symbol
from research_engine.ingest.parser import event_times
from research_engine.ingest.quality import inspect_recording
from research_engine.ingest.reader import iter_events
from research_engine.ingest.normalize import normalize_recording


def test_option_symbol_parser():
    contract = parse_option_symbol("QQQ260915C706000.US")
    assert contract.underlying == "QQQ"
    assert contract.expiry == date(2026, 9, 15)
    assert contract.call_put == "CALL"
    assert contract.strike == 706.0
    assert parse_option_symbol("QQQ.US") is None
    assert parse_option_symbol("bad") is None


def test_available_at_is_later_of_event_and_receive_time():
    payload = {
        "kind": "trade",
        "ts": "2026-09-15T13:20:44+00:00",
        "received_at": "2026-09-15T13:21:20+00:00",
    }
    event, received, available, source = event_times(payload)
    assert available == received
    assert available > event
    assert source == "event"


def test_depth_timestamp_falls_back_to_received():
    payload = {"kind": "depth", "received_at": "2026-09-15T13:20:54+00:00"}
    event, received, available, source = event_times(payload)
    assert event == received == available
    assert source == "received_fallback"


def test_reader_dedup_is_opt_in(tmp_path):
    path = tmp_path / "tiny.jsonl.gz"
    row = {"kind": "trade", "symbol": "QQQ.US", "ts": "2026-01-01T00:00:00Z"}
    with gzip.open(str(path), "wt") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.write(json.dumps(dict(reversed(list(row.items())))) + "\n")
    assert len(list(iter_events(path))) == 2
    assert len(list(iter_events(path, dedup_mode="exact_event"))) == 1


def test_quality_reports_duplicates_without_dropping(tmp_path):
    path = tmp_path / "tiny.jsonl.gz"
    rows = [
        {"kind": "trade", "symbol": "QQQ.US", "ts": "2026-01-02T14:30:00Z", "received_at": "2026-01-02T14:30:00Z", "price": 1, "volume": 2, "direction": "TradeDirection.Up"},
        {"kind": "trade", "symbol": "QQQ.US", "ts": "2026-01-02T14:30:00Z", "received_at": "2026-01-02T14:30:00Z", "price": 1, "volume": 2, "direction": "TradeDirection.Up"},
        {"kind": "depth", "symbol": "QQQ.US", "received_at": "2026-01-02T14:30:01Z", "bids": [[1, 1.1, 2, 0]], "asks": [[1, 1.0, 2, 0]]},
    ]
    with gzip.open(str(path), "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = inspect_recording(path)
    assert report.event_count == 3
    assert report.suspected_duplicate_count == 1
    assert report.invalid_spread == 1
    assert report.depth_event_timestamp_fallback == 1
    combined = inspect_recording([path, path])
    assert combined.event_count == 6
    assert combined.suspected_duplicate_count == 3


def test_normalized_provenance_is_portable_and_same_name_content_changes_invalidate_cache(tmp_path):
    source = tmp_path / 'sample.jsonl.gz'
    output = tmp_path / 'events.parquet'
    def write(price):
        with gzip.open(str(source), 'wt') as handle:
            handle.write(json.dumps(dict(kind='trade', symbol='QQQ.US', price=price,
                                         received_at='2026-09-17T14:00:00Z'))+'\n')
    write(100.)
    initial = normalize_recording(source, output, {})
    metadata = json.loads(output.with_suffix('.parquet.meta.json').read_text())
    assert metadata['source'] == source.name and metadata['output'] == output.name
    assert str(tmp_path) not in output.with_suffix('.parquet.meta.json').read_text()
    assert initial['output'] == str(output)
    assert normalize_recording(source, output, {})['cached']
    write(101.)
    changed = normalize_recording(source, output, {})
    assert not changed['cached'] and changed['source_hash'] != initial['source_hash']
