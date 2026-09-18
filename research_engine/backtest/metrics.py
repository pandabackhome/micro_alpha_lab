from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_metrics(trades: pd.DataFrame):
    if trades.empty:
        return {"trade_count": 0, "skipped_entries": trades.attrs.get("skipped_entries", 0),
                "skipped_late_entries": trades.attrs.get("skipped_late_entries", 0),
                "unclosed_positions": trades.attrs.get("unclosed_positions", 0)}, pd.DataFrame(columns=["date", "daily_pnl", "daily_trade_count", "daily_win_rate"])
    pnl = trades["pnl"]
    returns = trades["return"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity.to_numpy()])[1:]
    metrics = {
        "trade_count": len(trades),
        "skipped_entries": trades.attrs.get("skipped_entries", 0),
        "skipped_late_entries": trades.attrs.get("skipped_late_entries", 0),
        "unclosed_positions": trades.attrs.get("unclosed_positions", 0),
        "win_rate": float((pnl > 0).mean()),
        "loss_rate": float((pnl < 0).mean()), "average_return": float(returns.mean()),
        "median_return": float(returns.median()), "average_win": float(wins.mean()) if len(wins) else None,
        "average_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else None,
        "max_drawdown": float(drawdown.min()),
        "sharpe_like": float(returns.mean() / returns.std()) if returns.std() > 0 else None,
        "expectancy": float(pnl.mean()), "total_pnl": float(pnl.sum()),
        "mfe": float(trades["mfe"].mean()), "mae": float(trades["mae"].mean()),
        "average_holding_seconds": float(trades["holding_seconds"].mean()),
    }
    work = trades.copy()
    work["date"] = pd.to_datetime(work["entry_time"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    daily = work.groupby("date").agg(daily_pnl=("pnl", "sum"), daily_trade_count=("pnl", "size"),
                                     daily_win_rate=("pnl", lambda value: (value > 0).mean())).reset_index()
    return metrics, daily
