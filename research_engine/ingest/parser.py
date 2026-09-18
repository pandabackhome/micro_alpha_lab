from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


UTC = timezone.utc


def parse_timestamp(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def event_times(payload: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime], str]:
    """Return (event_ts, received_at, available_at, timestamp_source).

    A live strategy cannot observe a delayed event before it is received.
    ``available_at=max(event_ts, received_at)`` therefore enforces both
    event-time and receive-time causality.  Depth has no venue timestamp in
    the source schema and explicitly falls back to receive time.
    """
    received = parse_timestamp(payload.get("received_at"))
    event = parse_timestamp(payload.get("ts"))
    source = "event"
    if event is None and payload.get("kind") == "depth":
        event = received
        source = "received_fallback"
    available = max(x for x in (event, received) if x is not None) if event or received else None
    return event, received, available, source


def best_level(levels: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(levels, list):
        return None, None
    valid = []
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 3:
            continue
        price, size = level[1], level[2]
        if price is not None and size is not None and float(price) > 0 and float(size) >= 0:
            valid.append((int(level[0]), float(price), float(size)))
    if not valid:
        return None, None
    _, price, size = min(valid, key=lambda item: item[0])
    return price, size

