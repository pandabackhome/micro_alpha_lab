from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from research_engine.models import OptionContract


OPTION_RE = re.compile(
    r"^(?P<underlying>[A-Z][A-Z0-9]{0,9})"
    # Signalforge stores the OCC millistrike without mandatory zero padding
    # (706000 rather than 00706000), hence the intentionally variable width.
    r"(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{1,8})\.US$"
)


def parse_option_symbol(symbol: str) -> Optional[OptionContract]:
    match = OPTION_RE.fullmatch(symbol)
    if not match:
        return None
    expiry = datetime.strptime(match.group("expiry"), "%y%m%d").date()
    strike = int(match.group("strike")) / 1000.0
    right = match.group("right")
    return OptionContract(
        symbol=symbol,
        underlying=match.group("underlying"),
        expiry=expiry,
        call_put="CALL" if right == "C" else "PUT",
        strike=strike,
    )
