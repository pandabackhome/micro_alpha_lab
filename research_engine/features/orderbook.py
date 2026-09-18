from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from research_engine.features.snapshot import trailing_change, trailing_sum


def depth_imbalance(bid_size, ask_size) -> np.ndarray:
    bid, ask = np.asarray(bid_size, dtype=float), np.asarray(ask_size, dtype=float)
    denominator = bid + ask
    return np.divide(bid - ask, denominator, out=np.full_like(denominator, np.nan), where=denominator != 0)


def microprice(bid_price, ask_price, bid_size, ask_size) -> np.ndarray:
    bid, ask = np.asarray(bid_price, dtype=float), np.asarray(ask_price, dtype=float)
    bs, ass = np.asarray(bid_size, dtype=float), np.asarray(ask_size, dtype=float)
    denominator = bs + ass
    return np.divide(ass * bid + bs * ask, denominator, out=np.full_like(denominator, np.nan), where=denominator != 0)


def ofi_series(bid_price, bid_size, ask_price, ask_size) -> np.ndarray:
    """Cont-style L1 OFI computed from consecutive book states.

    e_t = I(Pb_t >= Pb_prev)Qb_t - I(Pb_t <= Pb_prev)Qb_prev
        - I(Pa_t <= Pa_prev)Qa_t + I(Pa_t >= Pa_prev)Qa_prev.
    Equal prices therefore reduce to size deltas. Missing/invalid pairs yield NaN.
    """
    bp, bs = np.asarray(bid_price, float), np.asarray(bid_size, float)
    ap, ass = np.asarray(ask_price, float), np.asarray(ask_size, float)
    result = np.full(len(bp), np.nan)
    valid = ~(np.isnan(bp[1:]) | np.isnan(bs[1:]) | np.isnan(ap[1:]) | np.isnan(ass[1:]) |
              np.isnan(bp[:-1]) | np.isnan(bs[:-1]) | np.isnan(ap[:-1]) | np.isnan(ass[:-1]))
    bid_part = np.where(bp[1:] > bp[:-1], bs[1:], np.where(bp[1:] < bp[:-1], -bs[:-1], bs[1:] - bs[:-1]))
    ask_part = np.where(ap[1:] < ap[:-1], -ass[1:], np.where(ap[1:] > ap[:-1], ass[:-1], ass[:-1] - ass[1:]))
    result[1:][valid] = (bid_part + ask_part)[valid]
    return result


def orderbook_features(bid_price, ask_price, bid_size, ask_size, windows: Sequence[int]) -> Dict[str, np.ndarray]:
    bid, ask = np.asarray(bid_price, float), np.asarray(ask_price, float)
    mid = (bid + ask) / 2.0
    spread = ask - bid
    imbalance = depth_imbalance(bid_size, ask_size)
    micro = microprice(bid, ask, bid_size, ask_size)
    ofi = ofi_series(bid, bid_size, ask, ask_size)
    output = {
        "qqq_bid": bid,
        "qqq_ask": ask,
        "qqq_mid": mid,
        "qqq_spread": spread,
        "qqq_spread_bps": np.divide(spread * 10000.0, mid, out=np.full_like(mid, np.nan), where=mid != 0),
        "bid_size": np.asarray(bid_size, float),
        "ask_size": np.asarray(ask_size, float),
        "depth_imbalance": imbalance,
        "depth_imbalance_change": trailing_change(imbalance),
        "microprice": micro,
        "microprice_minus_mid": micro - mid,
        "microprice_delta_bps": np.divide((micro - mid) * 10000.0, mid, out=np.full_like(mid, np.nan), where=mid != 0),
        "ofi": ofi,
    }
    for window in windows:
        output["depth_imbalance_{}s_avg".format(window)] = pd.Series(imbalance).rolling(window, min_periods=1).mean().to_numpy()
        output["ofi_{}s".format(window)] = trailing_sum(ofi, window)
    change = trailing_change(np.nan_to_num(ofi, nan=0.0))
    output["ofi_change"] = change
    output["ofi_acceleration"] = trailing_change(change)
    return output

