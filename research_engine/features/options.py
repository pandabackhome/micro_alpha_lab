from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from research_engine.features.orderbook import depth_imbalance
from research_engine.features.snapshot import bucket_numeric, causal_asof, datetime_ns, trailing_sum
from research_engine.features.trade_flow import trade_flow_features


def relative_name(offset: int) -> str:
    return "atm" if offset == 0 else ("p{}".format(offset) if offset > 0 else "m{}".format(abs(offset)))


def _target_strikes(spot: np.ndarray, strikes: np.ndarray, offset: int) -> np.ndarray:
    target = np.full(len(spot), np.nan)
    valid = ~np.isnan(spot)
    if not len(strikes) or not valid.any():
        return target
    positions = np.searchsorted(strikes, spot[valid])
    positions = np.clip(positions, 0, len(strikes) - 1)
    previous = np.maximum(positions - 1, 0)
    choose_previous = np.abs(strikes[previous] - spot[valid]) <= np.abs(strikes[positions] - spot[valid])
    nearest = np.where(choose_previous, previous, positions)
    shifted = nearest + offset
    spacing = float(np.median(np.diff(strikes))) if len(strikes) > 1 else 1.0
    near_real_atm = np.abs(strikes[nearest] - spot[valid]) <= spacing / 2 + 1e-6
    in_chain = near_real_atm & (shifted >= 0) & (shifted < len(strikes))
    values = np.full(valid.sum(), np.nan)
    values[in_chain] = strikes[shifted[in_chain]]
    target[valid] = values
    return target


def _empty(length: int) -> np.ndarray:
    return np.full(length, np.nan)


def mapped_option_features(
    grid: pd.DatetimeIndex,
    spot: np.ndarray,
    quotes: pd.DataFrame,
    depths: pd.DataFrame,
    trades: pd.DataFrame,
    relative_strikes: Sequence[int],
    return_windows: Sequence[int],
    volume_windows: Sequence[int],
    flow_window: int,
    burst_lookback: int,
) -> Dict[str, np.ndarray]:
    strikes = np.array(sorted(set(quotes["option_strike"].dropna()) | set(depths["option_strike"].dropna()) |
                              set(trades["option_strike"].dropna())), dtype=float)
    quote_groups = {key: value for key, value in quotes.groupby("symbol", sort=False)}
    depth_groups = {key: value for key, value in depths.groupby("symbol", sort=False)}
    trade_groups = {key: value for key, value in trades.groupby("symbol", sort=False)}
    symbols: Dict[Tuple[str, float], str] = {}
    for frame in (quotes, depths, trades):
        if frame.empty:
            continue
        for row in frame[["symbol", "option_right", "option_strike"]].drop_duplicates().itertuples(index=False):
            symbols[(row.option_right, float(row.option_strike))] = row.symbol

    state_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def state(symbol: str) -> Dict[str, np.ndarray]:
        if symbol in state_cache:
            return state_cache[symbol]
        q = quote_groups.get(symbol)
        d = depth_groups.get(symbol)
        t = trade_groups.get(symbol)
        result = {name: _empty(len(grid)) for name in ("last", "bid", "ask", "bid_size", "ask_size")}
        if q is not None and not q.empty:
            result["last"] = causal_asof(grid, q["available_at"], {"v": q["price"]})["v"]
        if d is not None and not d.empty:
            book = causal_asof(grid, d["available_at"], {
                "bid": d["best_bid"], "ask": d["best_ask"],
                "bid_size": d["best_bid_size"], "ask_size": d["best_ask_size"],
                "book_time_ns": datetime_ns(d["available_at"]),
            })
            book["depth_age_seconds"] = (grid.astype("int64").to_numpy() - book.pop("book_time_ns")) / 1e9
            result.update(book)
        else:
            result["depth_age_seconds"] = _empty(len(grid))
        if t is not None and not t.empty:
            flow = trade_flow_features(
                grid, t["available_at"], t["volume"], t["direction"],
                sorted(set(volume_windows) | {flow_window}), prefix="x_",
            )
            for window in volume_windows:
                result["volume_{}s".format(window)] = flow["x_total_volume_{}s".format(window)]
            result["trade_count"] = flow["x_trade_count_{}s".format(flow_window)]
            result["trade_imbalance"] = flow["x_trade_imbalance_{}s".format(flow_window)]
        else:
            for window in volume_windows:
                result["volume_{}s".format(window)] = np.zeros(len(grid))
            result["trade_count"] = np.zeros(len(grid))
            result["trade_imbalance"] = np.zeros(len(grid))
        burst_window = 5 if 5 in volume_windows else volume_windows[0]
        volume = pd.Series(result["volume_{}s".format(burst_window)])
        baseline = volume.rolling(burst_lookback, min_periods=max(10, burst_lookback // 10)).median()
        result["volume_burst"] = (volume / baseline.replace(0, np.nan)).to_numpy()
        state_cache[symbol] = result
        return result

    output: Dict[str, np.ndarray] = {}
    for right in ("CALL", "PUT"):
        for offset in relative_strikes:
            base = "{}_{}_".format(right.lower(), relative_name(offset))
            target = _target_strikes(spot, strikes, int(offset))
            selected = {name: _empty(len(grid)) for name in (
                "option_last", "option_bid", "option_ask", "option_bid_size", "option_ask_size",
                "option_trade_count", "option_trade_imbalance", "option_depth_age_seconds",
                "option_volume_burst",
            )}
            for window in volume_windows:
                selected["option_volume_{}s".format(window)] = np.zeros(len(grid))
            for strike in np.unique(target[~np.isnan(target)]):
                mask = target == strike
                symbol = symbols.get((right, float(strike)))
                if not symbol:
                    continue
                values = state(symbol)
                for destination, source in (
                    ("option_last", "last"), ("option_bid", "bid"), ("option_ask", "ask"),
                    ("option_bid_size", "bid_size"), ("option_ask_size", "ask_size"),
                    ("option_trade_count", "trade_count"), ("option_trade_imbalance", "trade_imbalance"),
                    ("option_depth_age_seconds", "depth_age_seconds"),
                    ("option_volume_burst", "volume_burst"),
                ):
                    selected[destination][mask] = values[source][mask]
                for window in volume_windows:
                    name = "option_volume_{}s".format(window)
                    selected[name][mask] = values["volume_{}s".format(window)][mask]
            selected["option_strike"] = target
            selected["strike_distance"] = np.divide(target - spot, spot, out=_empty(len(grid)), where=spot != 0)
            selected["option_mid"] = (selected["option_bid"] + selected["option_ask"]) / 2.0
            selected["option_spread"] = selected["option_ask"] - selected["option_bid"]
            selected["option_spread_pct"] = np.divide(
                selected["option_spread"], selected["option_mid"], out=_empty(len(grid)),
                where=selected["option_mid"] != 0,
            )
            selected["option_depth_imbalance"] = depth_imbalance(
                selected["option_bid_size"], selected["option_ask_size"]
            )
            mid_series = pd.Series(selected["option_mid"])
            for window in return_windows:
                same_contract = pd.Series(target).eq(pd.Series(target).shift(window))
                selected["option_return_{}s".format(window)] = np.log(
                    mid_series / mid_series.shift(window)
                ).where(same_contract).to_numpy()
            for name, values in selected.items():
                output[base + name] = values
    return output


def chain_flow_features(
    grid: pd.DatetimeIndex,
    spot: np.ndarray,
    trades: pd.DataFrame,
    window: int,
    near_atm_distance: float,
) -> Dict[str, np.ndarray]:
    if trades.empty:
        return {name: np.zeros(len(grid)) for name in (
            "call_buy_volume", "call_sell_volume", "put_buy_volume", "put_sell_volume",
            "call_trade_imbalance", "put_trade_imbalance", "call_volume", "put_volume",
            "call_put_volume_ratio", "near_atm_call_flow", "near_atm_put_flow", "otm_call_flow", "otm_put_flow",
        )}
    grid_ns = grid.astype("int64").to_numpy()
    event_ns = datetime_ns(trades["available_at"])
    bucket = np.searchsorted(grid_ns, event_ns, side="left")
    valid = (bucket >= 0) & (bucket < len(grid))
    # Assign trades to the *next* snapshot bucket for volume availability,
    # but compute strike distance using the previous completed snapshot. The
    # next snapshot spot may contain a quote that arrives after the trade.
    trade_spot = np.full(len(trades), np.nan)
    spot_index = np.searchsorted(grid_ns, event_ns, side="right") - 1
    spot_valid = valid & (spot_index >= 0)
    trade_spot[spot_valid] = spot[spot_index[spot_valid]]
    strike = trades["option_strike"].to_numpy(dtype=float)
    distance = np.divide(strike - trade_spot, trade_spot, out=np.full(len(trades), np.nan), where=trade_spot != 0)
    right = trades["option_right"].to_numpy(object)
    direction = trades["direction"].to_numpy(object)
    volume = trades["volume"].fillna(0).to_numpy(float)

    def roll(mask):
        return trailing_sum(bucket_numeric(grid, trades["available_at"], np.where(mask, volume, 0.0)), window)

    cb = roll((right == "CALL") & (direction == "TradeDirection.Up"))
    cs = roll((right == "CALL") & (direction == "TradeDirection.Down"))
    pb = roll((right == "PUT") & (direction == "TradeDirection.Up"))
    ps = roll((right == "PUT") & (direction == "TradeDirection.Down"))
    cv, pv = roll(right == "CALL"), roll(right == "PUT")
    directional_call, directional_put = cb + cs, pb + ps
    near = np.abs(distance) < near_atm_distance
    signed = np.where(direction == "TradeDirection.Up", volume,
                      np.where(direction == "TradeDirection.Down", -volume, 0.0))
    near_call = trailing_sum(bucket_numeric(grid, trades["available_at"], np.where(near & (right == "CALL"), signed, 0)), window)
    near_put = trailing_sum(bucket_numeric(grid, trades["available_at"], np.where(near & (right == "PUT"), signed, 0)), window)
    otm_call = trailing_sum(bucket_numeric(grid, trades["available_at"], np.where((distance > 0) & (right == "CALL"), signed, 0)), window)
    otm_put = trailing_sum(bucket_numeric(grid, trades["available_at"], np.where((distance < 0) & (right == "PUT"), signed, 0)), window)
    return {
        "call_buy_volume": cb, "call_sell_volume": cs, "put_buy_volume": pb, "put_sell_volume": ps,
        "call_trade_imbalance": np.divide(cb - cs, directional_call, out=np.zeros_like(cv), where=directional_call != 0),
        "put_trade_imbalance": np.divide(pb - ps, directional_put, out=np.zeros_like(pv), where=directional_put != 0),
        "call_volume": cv, "put_volume": pv,
        "call_put_volume_ratio": np.divide(cv, pv, out=np.full_like(cv, np.nan), where=pv != 0),
        "near_atm_call_flow": near_call, "near_atm_put_flow": near_put,
        "otm_call_flow": otm_call, "otm_put_flow": otm_put,
    }
