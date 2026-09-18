from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def fill_price(row, side: str, action: str, mode: str, config: Dict,
               contract: Optional[str] = None) -> Optional[Tuple[float, str, int]]:
    """Ask to buy, bid to sell; quote spread is already embedded in fills."""
    if mode == "underlying":
        symbol, multiplier = "QQQ.US", 1
        quote_side = "qqq_ask" if (side == "LONG") == (action == "entry") else "qqq_bid"
        age_column = "qqq_depth_age_seconds"
    elif mode == "option":
        right = "call_" if side == "LONG" else "put_"
        prefix = right + "atm_"
        if contract is not None:
            # Search the mapped relative-strike slots for the *entry* contract
            # rather than silently rolling an open position into a new ATM.
            for key in row.index:
                if key.startswith(right) and key.endswith("_option_strike"):
                    value = row[key]
                    if value is not None and np.isfinite(value) and str(value) == contract:
                        prefix = key[:-len("option_strike")]
                        break
            else:
                return None
        multiplier = int(config["execution"].get("option_multiplier", 100))
        quote_side = prefix + ("option_ask" if action == "entry" else "option_bid")
        age_column = prefix + "option_depth_age_seconds"
        symbol = row.get(prefix + "option_strike")
    else:
        raise ValueError("execution mode must be underlying or option")
    age = row.get(age_column)
    if age is not None and (not np.isfinite(age) or age > float(config["execution"].get("max_quote_age_seconds", 5))):
        return None
    price = row.get(quote_side)
    if price is None or not np.isfinite(price) or price <= 0:
        return None
    slippage = float(config["execution"].get("slippage_bps", 0)) / 10000.0
    is_buy = action == "entry" or (action == "exit" and mode == "underlying" and side == "SHORT")
    if mode == "underlying" and side == "SHORT":
        is_buy = action == "exit"
    if mode == "option":
        is_buy = action == "entry"
    return price * (1.0 + slippage if is_buy else 1.0 - slippage), str(symbol), multiplier
