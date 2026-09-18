from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, TextIO, Union

from research_engine.models import RawEvent


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(str(path), mode="rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_events(
    path: Union[str, Path],
    *,
    dedup_mode: str = "none",
    include_metadata: bool = True,
) -> Iterator[RawEvent]:
    """Stream JSONL/JSONL.GZ without loading the recording into memory.

    ``exact_event`` is deliberately opt-in.  In the absence of an exchange
    trade id or sequence id, the default is to preserve apparently identical
    prints because they may be distinct executions.
    """
    if dedup_mode not in {"none", "exact_event"}:
        raise ValueError("dedup_mode must be 'none' or 'exact_event'")
    source = Path(path)
    seen = set()
    with _open_text(source) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not include_metadata and payload.get("kind") in {"header", "contracts"}:
                continue
            if dedup_mode == "exact_event":
                fingerprint = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
            yield RawEvent(line_number=line_number, payload=payload)


def read_metadata(path: Union[str, Path]) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    for event in iter_events(path):
        if event.kind in {"header", "contracts"}:
            result[event.kind] = event.payload
        if len(result) == 2:
            break
    return result
