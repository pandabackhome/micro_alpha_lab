from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from research_engine.backtest.execution import fill_price


def backtest_day(features: pd.DataFrame, signals: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Non-overlapping positions; enter at the next grid row after signal.

    Every exit uses the next observed executable side of the book. Option
    exits must use the entry contract, not the new ATM contract after a spot
    move. Current snapshots expose only relative contracts, so positions
    terminate if that contract leaves the relative-strike window.
    """
    if features.empty or signals.empty:
        return pd.DataFrame()
    mode = config["execution"]["mode"]
    frame = features.sort_values("timestamp").reset_index(drop=True)
    stamps = pd.DatetimeIndex(frame["timestamp"])
    signal_times = pd.DatetimeIndex(signals["timestamp"])
    signal_indices = stamps.searchsorted(signal_times, side="right")
    rows = []
    occupied_until = -1
    skipped_entries = skipped_late_entries = unclosed_positions = 0
    unclosed_records = []
    max_hold = int(config["exit"]["max_holding_seconds"])
    stop = float(config["exit"].get("stop_loss_pct", -1))
    take = float(config["exit"].get("take_profit_pct", 10))
    trailing = config["exit"].get("trailing_stop_pct")
    commission = float(config["execution"].get("commission_per_contract", 0)) if mode == "option" else 0.0
    for signal, entry_index in zip(signals.itertuples(index=False), signal_indices):
        if entry_index >= len(frame) or entry_index <= occupied_until:
            continue
        side = signal.side
        entry_row = frame.iloc[entry_index]
        if entry_row["timestamp"] + pd.Timedelta(seconds=max_hold) > frame.iloc[-1]["timestamp"]:
            skipped_late_entries += 1
            continue
        filled = fill_price(entry_row, side, "entry", mode, config)
        if filled is None:
            skipped_entries += 1
            continue
        entry_price, contract, multiplier = filled
        peak = 0.0
        mfe, mae = -np.inf, np.inf
        limit_time = entry_row["timestamp"] + pd.Timedelta(seconds=max_hold)
        expiry_index = int(stamps.searchsorted(limit_time, side="left"))
        # Wait for the next genuinely available executable quote after the
        # deadline; never reuse a stale quote from before the deadline.
        exit_deadline = limit_time + pd.Timedelta(seconds=int(config["exit"].get("max_exit_delay_seconds", 30)))
        search_end = min(len(frame) - 1, int(stamps.searchsorted(exit_deadline, side="right")) - 1)
        exit_index = search_end
        exit_reason = "max_holding"
        exit_fill = None
        for index in range(entry_index + 1, search_end + 1):
            row = frame.iloc[index]
            fill = fill_price(row, side, "exit", mode, config, contract=contract if mode == "option" else None)
            if fill is None or (mode == "option" and fill[1] != contract):
                continue
            exit_price = fill[0]
            gross_pct = (exit_price / entry_price - 1) * (1 if side == "LONG" or mode == "option" else -1)
            mfe, mae = max(mfe, gross_pct), min(mae, gross_pct)
            peak = max(peak, gross_pct)
            if gross_pct <= stop:
                exit_reason = "stop_loss"
            elif gross_pct >= take:
                exit_reason = "take_profit"
            elif trailing is not None and gross_pct <= peak - float(trailing):
                exit_reason = "trailing_stop"
            elif index >= expiry_index:
                exit_reason = "max_holding"
            else:
                continue
            exit_index = index
            exit_fill = fill
            break
        if exit_fill is None:
            unclosed_positions += 1
            unclosed_records.append({
                "signal_time": signal.timestamp, "entry_time": entry_row["timestamp"],
                "side": side, "symbol": contract, "entry_price": entry_price,
                "last_snapshot_time": frame.iloc[search_end]["timestamp"],
                "reason": "no fresh executable exit quote within deadline",
            })
            # Position has not been liquidated; do not open another overlapping
            # trade later in the session under an imaginary flat book.
            occupied_until = len(frame) - 1
            continue
        exit_price = exit_fill[0]
        pnl = (exit_price - entry_price) * multiplier * (1 if side == "LONG" or mode == "option" else -1) - 2 * commission
        cost = entry_price * multiplier
        rows.append({
            "signal_time": signal.timestamp, "entry_time": entry_row["timestamp"],
            "exit_time": frame.iloc[exit_index]["timestamp"], "side": side,
            "symbol": contract, "entry_price": entry_price, "exit_price": exit_price,
            "multiplier": multiplier, "pnl": pnl, "return": pnl / cost,
            "mfe": mfe if np.isfinite(mfe) else np.nan, "mae": mae if np.isfinite(mae) else np.nan,
            "holding_seconds": (frame.iloc[exit_index]["timestamp"] - entry_row["timestamp"]).total_seconds(),
            "exit_reason": exit_reason,
        })
        occupied_until = exit_index
    result = pd.DataFrame(rows)
    result.attrs["skipped_entries"] = skipped_entries
    result.attrs["skipped_late_entries"] = skipped_late_entries
    result.attrs["unclosed_positions"] = unclosed_positions
    result.attrs["unclosed_records"] = unclosed_records
    return result
