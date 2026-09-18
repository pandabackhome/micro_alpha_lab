from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    expiry: date
    call_put: str
    strike: float


@dataclass
class RawEvent:
    line_number: int
    payload: Dict[str, Any]

    @property
    def kind(self) -> str:
        return str(self.payload.get("kind", "unknown"))


@dataclass
class QualityReport:
    source: str
    event_count: int = 0
    counts_by_kind: Dict[str, int] = field(default_factory=dict)
    qqq_event_count: int = 0
    option_event_count: int = 0
    first_event_timestamp: Optional[str] = None
    last_event_timestamp: Optional[str] = None
    first_received_timestamp: Optional[str] = None
    last_received_timestamp: Optional[str] = None
    suspected_duplicate_count: int = 0
    suspected_duplicate_ratio: float = 0.0
    out_of_order_event_count: int = 0
    out_of_order_ratio: float = 0.0
    out_of_order_received_count: int = 0
    out_of_order_received_ratio: float = 0.0
    missing_timestamp: int = 0
    depth_event_timestamp_fallback: int = 0
    invalid_price: int = 0
    invalid_spread: int = 0
    malformed_json: int = 0
    symbol_count: int = 0
    option_strike_count: int = 0
    symbols: List[str] = field(default_factory=list)
    option_strikes: List[float] = field(default_factory=list)
    event_lag_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

