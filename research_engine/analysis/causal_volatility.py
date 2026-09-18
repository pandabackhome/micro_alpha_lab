"""Historical, causal QQQ volatility regimes and fixed-contract straddle fills.

Run with ``python -m research_engine.analysis.causal_volatility --data-root PATH``.
All high/low thresholds use previous dates; an exit never selects a new ATM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from importlib.metadata import version
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from research_engine.config import load_config, resolve_path
from research_engine.features.snapshot import datetime_ns
from research_engine.ingest.option_symbol import parse_option_symbol


@dataclass(frozen=True)
class StudySettings:
    calibration_days: int = 5
    min_history_samples: int = 100
    volatility_seconds: int = 30
    signal_seconds: int = 60
    latency_seconds: int = 1
    holding_seconds: int = 30
    max_exit_delay_seconds: int = 30
    max_quote_age_seconds: float = 5.0
    min_entry_mid: float = 0.05
    max_atm_distance: float = 0.5
    commission_per_contract: float = 0.65
    slippage_bps: float = 1.0
    multiplier: int = 100

    def __post_init__(self):
        for name in ("calibration_days", "min_history_samples", "volatility_seconds",
                     "signal_seconds", "latency_seconds", "holding_seconds", "multiplier"):
            if getattr(self, name) < 1:
                raise ValueError(name + " must be positive")
        if self.signal_seconds % self.volatility_seconds:
            raise ValueError("signal_seconds must be a multiple of volatility_seconds")
        for name in ("max_exit_delay_seconds", "max_quote_age_seconds", "min_entry_mid",
                     "max_atm_distance", "commission_per_contract", "slippage_bps"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.slippage_bps >= 10000:
            raise ValueError("slippage_bps must be less than 10000")


FEATURE_COLUMNS = ["timestamp", "qqq_bid", "qqq_ask", "qqq_depth_age_seconds"]
BOOK_COLUMNS = ["symbol", "available_at", "best_bid", "best_ask", "best_bid_size",
                "best_ask_size", "option_right", "option_strike"]
GROUPS = ("low", "middle", "high")


def observations(frame: pd.DataFrame, settings: StudySettings, sampling_seconds=None) -> pd.DataFrame:
    """Recompute trailing volatility from fresh spot mids on a complete 1s grid."""
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)).as_unit("ns")
    if len(stamps) < 2 or not np.all(np.diff(stamps.asi8) == 1_000_000_000):
        raise ValueError("study requires a complete, unique 1-second feature grid")
    local = stamps.tz_convert("America/New_York")
    if len(set(local.date)) != 1:
        raise ValueError("each feature file must contain one New York date")
    day = str(local.date[0])
    opened = pd.Timestamp(day + " 09:30", tz="America/New_York").tz_convert("UTC")
    closed = pd.Timestamp(day + " 16:00", tz="America/New_York").tz_convert("UTC")
    age = frame["qqq_depth_age_seconds"]
    bid, ask = frame["qqq_bid"], frame["qqq_ask"]
    good = (age.between(0, settings.max_quote_age_seconds) & bid.gt(0) & ask.ge(bid) &
            np.isfinite(bid) & np.isfinite(ask))
    mid = ((bid + ask) / 2).where(good)
    returns = np.log(mid / mid.shift(1))
    vol = returns.pow(2).rolling(settings.volatility_seconds,
                                min_periods=settings.volatility_seconds).sum().pow(0.5)
    seconds = (stamps - opened).total_seconds().astype(int)
    step = settings.volatility_seconds if sampling_seconds is None else sampling_seconds
    if not isinstance(step, int) or step < 1 or settings.signal_seconds % step:
        raise ValueError("sampling seconds must be a positive divisor of the signal interval")
    chosen = ((seconds >= settings.volatility_seconds) & (stamps < closed) &
              (seconds % step == 0))
    result = pd.DataFrame({
        "timestamp": stamps, "date": day, "half_hour": seconds // 1800,
        "spot": mid, "past_vol": vol, "session_close": closed,
        "scheduled": (seconds - settings.volatility_seconds) % settings.signal_seconds == 0,
    }).loc[chosen].reset_index(drop=True)
    return result


def assign_regimes(current: pd.DataFrame, history: List[pd.DataFrame],
                   settings: StudySettings) -> pd.DataFrame:
    """Freeze each day's thresholds before seeing any of that day's outcomes."""
    result = current.copy()
    result["regime"] = "warmup"
    result["q25"] = np.nan
    result["q75"] = np.nan
    result["history_samples"] = 0
    result["history_first_date"] = None
    result["history_last_date"] = None
    selected = history[-settings.calibration_days:]
    if selected:
        past = pd.concat(selected, ignore_index=True)
        if past["date"].max() >= current["date"].min():
            raise ValueError("threshold history must contain only earlier trading dates")
        if past["date"].nunique() != len(selected):
            raise ValueError("threshold history must contain distinct dates")
        result["history_first_date"] = past["date"].min()
        result["history_last_date"] = past["date"].max()
        if len(selected) == settings.calibration_days:
            for bucket, indices in result.groupby("half_hour").groups.items():
                values = past.loc[past["half_hour"].eq(bucket), "past_vol"].dropna()
                result.loc[indices, "history_samples"] = len(values)
                if len(values) < settings.min_history_samples:
                    result.loc[indices, "regime"] = "insufficient_history"
                    continue
                q25, q75 = values.quantile([0.25, 0.75]).to_numpy()
                result.loc[indices, ["q25", "q75"]] = (q25, q75)
                vol = result.loc[indices, "past_vol"]
                # Equal thresholds produce no high/low signal, preserving ties.
                regime = np.where(vol < q25, "low", np.where(vol > q75, "high", "middle"))
                result.loc[indices, "regime"] = regime
    result.loc[result["past_vol"].isna(), "regime"] = "invalid_spot"
    return result


class ContractBooks:
    """As-of quote lookup that retains invalid updates instead of hiding them."""

    def __init__(self, events: pd.DataFrame):
        self.books = {}
        pairs = {}
        for symbol, group in events.dropna(subset=["available_at"]).groupby("symbol", sort=True):
            group = group.sort_values("available_at", kind="stable")
            times = datetime_ns(group["available_at"])
            values = group[["best_bid", "best_ask", "best_bid_size", "best_ask_size"]].to_numpy(float)
            self.books[symbol] = (times, values)
            right = group["option_right"].iloc[0]
            strike = float(group["option_strike"].iloc[0])
            key = (strike, right)
            if key in pairs:
                raise ValueError("multiple symbols share an expiry/right/strike")
            pairs[key] = (symbol, times[0])
        self.contracts = [(strike, right, symbol, first_seen)
                          for (strike, right), (symbol, first_seen) in pairs.items()]
        self.pairs = []
        for strike in sorted({key[0] for key in pairs}):
            call, put = pairs.get((strike, "CALL")), pairs.get((strike, "PUT"))
            if call and put:
                self.pairs.append((strike, call[0], put[0], max(call[1], put[1])))

    def select(self, when: pd.Timestamp, spot: float, max_distance: float):
        known = [pair for pair in self.pairs if pair[3] <= when.value]
        if not known:
            return None
        pair = min(known, key=lambda p: (abs(p[0] - spot), p[0]))
        return pair[:3] if abs(pair[0] - spot) <= max_distance + 1e-9 else None

    def select_leg(self, when: pd.Timestamp, spot: float, right: str, max_distance: float):
        """Select a known contract of the requested right without requiring its other leg."""
        known = [item for item in self.contracts if item[1] == right and item[3] <= when.value]
        if not known:
            return None
        item = min(known, key=lambda candidate: (abs(candidate[0] - spot), candidate[0]))
        return (item[0], item[2]) if abs(item[0] - spot) <= max_distance + 1e-9 else None

    def quote(self, symbol: str, when: pd.Timestamp, max_age: float):
        if symbol not in self.books:
            return None, "missing_contract"
        times, values = self.books[symbol]
        index = int(np.searchsorted(times, when.value, side="right")) - 1
        if index < 0:
            return None, "missing_quote"
        age = (when.value - times[index]) / 1e9
        if age > max_age:
            return None, "stale_quote"
        bid, ask, bid_size, ask_size = values[index]
        if not np.isfinite([bid, ask, bid_size, ask_size]).all():
            return None, "incomplete_quote"
        if bid <= 0 or ask < bid:
            return None, "invalid_price"
        if bid_size < 1 or ask_size < 1:
            return None, "insufficient_size"
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "age": age}, "ok"

    def pair_quote(self, call: str, put: str, when: pd.Timestamp, max_age: float):
        cq, cr = self.quote(call, when, max_age)
        pq_, pr = self.quote(put, when, max_age)
        if cq is None or pq_ is None:
            return None, "call:" + cr + ";put:" + pr
        return (cq, pq_), "ok"


def simulate_day(signals: pd.DataFrame, books: ContractBooks,
                 settings: StudySettings) -> pd.DataFrame:
    """One paper straddle at a time; retain every scheduled opportunity/status."""
    rows = []
    occupied_until = None
    for signal in signals.loc[signals["scheduled"]].sort_values("timestamp").to_dict("records"):
        row = dict(signal, status=signal["regime"], reason="")
        rows.append(row)
        if signal["regime"] not in GROUPS:
            continue
        when, close = signal["timestamp"], signal["session_close"]
        if occupied_until is not None and when < occupied_until:
            row.update(status="busy", reason="previous straddle remains open")
            continue
        entry_time = when + pd.Timedelta(seconds=settings.latency_seconds)
        deadline = entry_time + pd.Timedelta(seconds=settings.holding_seconds)
        if deadline > close:
            row.update(status="session_end", reason="holding period exceeds session")
            continue
        contract = books.select(when, signal["spot"], settings.max_atm_distance)
        if contract is None:
            row.update(status="missing_contract", reason="no observed ATM call/put pair at signal time")
            continue
        strike, call, put = contract
        row.update(strike=strike, call_symbol=call, put_symbol=put,
                   entry_time=entry_time, exit_due=deadline)
        entry, reason = books.pair_quote(call, put, entry_time, settings.max_quote_age_seconds)
        if entry is None:
            row.update(status="entry_unavailable", reason=reason)
            continue
        if min(q["mid"] for q in entry) < settings.min_entry_mid:
            row.update(status="entry_unavailable", reason="entry mid below configured minimum")
            continue
        slip = settings.slippage_bps / 10000
        entry_ask = sum(q["ask"] for q in entry)
        entry_fill = entry_ask * (1 + slip)
        entry_fee = 2 * settings.commission_per_contract
        row.update(entry_debit=entry_fill * settings.multiplier + entry_fee,
                   entry_mid=sum(q["mid"] for q in entry), entry_ask=entry_ask,
                   entry_fill=entry_fill, entry_fee=entry_fee,
                   entry_call_mid=entry[0]["mid"], entry_put_mid=entry[1]["mid"])
        limit = min(close, deadline + pd.Timedelta(seconds=settings.max_exit_delay_seconds))
        exit_quote = None
        missed = 0
        for exit_time in pd.date_range(deadline, limit, freq="1s"):
            exit_quote, reason = books.pair_quote(call, put, exit_time, settings.max_quote_age_seconds)
            if exit_quote is not None:
                break
            missed += 1
        row["missing_exit_checks"] = missed
        if exit_quote is None:
            row.update(status="unclosed", reason=reason, last_exit_check=limit)
            occupied_until = close
            continue
        exit_bid = sum(q["bid"] for q in exit_quote)
        exit_fill = exit_bid * (1 - slip)
        exit_mid = sum(q["mid"] for q in exit_quote)
        multiplier = settings.multiplier
        mid_pnl = (exit_mid - row["entry_mid"]) * multiplier
        spread_cost = ((entry_ask - row["entry_mid"]) + (exit_mid - exit_bid)) * multiplier
        slippage_cost = ((entry_fill - entry_ask) + (exit_bid - exit_fill)) * multiplier
        commission = 4 * settings.commission_per_contract
        net = mid_pnl - spread_cost - slippage_cost - commission
        row.update(status="closed", exit_time=exit_time, exit_mid=exit_mid,
                   exit_bid=exit_bid, exit_fill=exit_fill,
                   exit_call_mid=exit_quote[0]["mid"], exit_put_mid=exit_quote[1]["mid"],
                   exit_delay_seconds=(exit_time - deadline).total_seconds(),
                   mid_pnl=mid_pnl, spread_cost=spread_cost, slippage_cost=slippage_cost,
                   commission=commission, net_pnl=net, net_return=net / row["entry_debit"],
                   abs_option_mid_change=(abs(exit_quote[0]["mid"] - entry[0]["mid"]) +
                                          abs(exit_quote[1]["mid"] - entry[1]["mid"])) / 2)
        occupied_until = exit_time
    return pd.DataFrame(rows)


def summarize(ledger: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    summaries, daily = [], []
    for regime in GROUPS:
        group = ledger.loc[ledger["regime"].eq(regime)]
        done = group.loc[group["status"].eq("closed")]
        unclosed = group.loc[group["status"].eq("unclosed")]
        debit = float(unclosed["entry_debit"].sum()) if len(unclosed) else 0.0
        realized = float(done["net_pnl"].sum()) if len(done) else 0.0
        result = {"regime": regime, "signals": len(group), "closed": len(done),
                  "unclosed": len(unclosed), "unclosed_debit": debit,
                  "entry_unavailable": int(group["status"].eq("entry_unavailable").sum()),
                  "missing_contract": int(group["status"].eq("missing_contract").sum()),
                  "busy": int(group["status"].eq("busy").sum()),
                  "session_end": int(group["status"].eq("session_end").sum()),
                  "realized_net_pnl": realized,
                  "net_with_unclosed_zero_recovery": realized - debit,
                  "delayed_exits": int(done["exit_delay_seconds"].gt(0).sum()) if len(done) else 0}
        for metric in ("mid_pnl", "spread_cost", "slippage_cost", "commission", "net_pnl",
                       "net_return", "abs_option_mid_change"):
            result["mean_" + metric] = float(done[metric].mean()) if len(done) else None
        result["win_rate"] = float(done["net_pnl"].gt(0).mean()) if len(done) else None
        summaries.append(result)
        for day, records in group.groupby("date"):
            closed = records.loc[records["status"].eq("closed")]
            unresolved = records.loc[records["status"].eq("unclosed")]
            daily.append({"date": day, "regime": regime, "signals": len(records),
                          "closed": len(closed), "unclosed": len(unresolved),
                          "mean_net_pnl": closed["net_pnl"].mean() if len(closed) else np.nan,
                          "mean_abs_option_mid_change": closed["abs_option_mid_change"].mean() if len(closed) else np.nan})
    daily_frame = pd.DataFrame(daily)
    comparisons = {}
    if not daily_frame.empty:
        for metric in ("mean_net_pnl", "mean_abs_option_mid_change"):
            paired = daily_frame.pivot(index="date", columns="regime", values=metric)
            if not {"high", "low"}.issubset(paired.columns):
                continue
            diff = (paired["high"] - paired["low"]).dropna()
            if len(diff):
                rng = np.random.RandomState(42)
                samples = rng.choice(diff.to_numpy(), size=(10000, len(diff))).mean(axis=1)
                comparisons[metric] = {"paired_days": len(diff), "high_minus_low": float(diff.mean()),
                                       "positive_days": int(diff.gt(0).sum()),
                                       "daily_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    return pd.DataFrame(summaries), daily_frame, comparisons


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_books(path: Path, day: str) -> ContractBooks:
    events = pq.read_table(path, columns=BOOK_COLUMNS,
                           filters=[("kind", "=", "depth"),
                                    ("option_expiry", "=", date.fromisoformat(day))]).to_pandas()
    symbols = []
    for symbol in events["symbol"].dropna().unique():
        contract = parse_option_symbol(symbol)
        if contract and contract.underlying == "QQQ" and contract.expiry == date.fromisoformat(day):
            symbols.append(symbol)
    return ContractBooks(events.loc[events["symbol"].isin(symbols)])


def run_study(data_root: Path, output: Path, settings: StudySettings) -> Dict:
    paths = sorted((data_root / "features").glob("*.parquet"))
    if len(paths) <= settings.calibration_days:
        raise ValueError("need more feature dates than calibration_days")
    # Fail before creating partial artifacts if fixed-contract source books are absent.
    for path in paths[settings.calibration_days:]:
        normalized = data_root / "normalized" / path.stem / "events.parquet"
        if not normalized.exists():
            raise ValueError("missing full-contract normalized book: normalized/" + path.stem)
    histories, classified, ledgers, manifest = [], [], [], []
    for path in paths:
        current = observations(pd.read_parquet(path, columns=FEATURE_COLUMNS), settings)
        if current.empty or set(current["date"]) != {path.stem}:
            raise ValueError("feature filename/date mismatch or empty regular session")
        signals = assign_regimes(current, histories, settings)
        classified.append(signals)
        histories.append(current)
        item = {"file": "features/" + path.name, "sha256": file_hash(path)}
        manifest.append(item)
        if signals["regime"].isin(GROUPS).any():
            normalized = data_root / "normalized" / path.stem / "events.parquet"
            books = read_books(normalized, path.stem)
            manifest.append({"file": "normalized/" + path.stem + "/events.parquet",
                             "sha256": file_hash(normalized)})
        else:
            books = ContractBooks(pd.DataFrame(columns=BOOK_COLUMNS))
        ledger = simulate_day(signals, books, settings)
        ledgers.append(ledger)
        print(json.dumps({"date": path.stem, "status_counts": ledger["status"].value_counts().to_dict()}), flush=True)
    all_signals = pd.concat(classified, ignore_index=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    summary, daily, comparisons = summarize(ledger)
    metadata = {"study": "past_dates_volatility_fixed_contract_straddle",
                "validation": "retrospective_walk_forward_previously_explored_dates",
                "settings": asdict(settings), "dates": [p.stem for p in paths],
                "calibration_dates": [p.stem for p in paths[:settings.calibration_days]],
                "evaluation_dates": sorted(ledger.loc[ledger["regime"].isin(GROUPS), "date"].unique()),
                "status_counts": ledger["status"].value_counts().to_dict(),
                "paired_daily_comparisons": comparisons,
                "source_manifest": manifest, "study_code_sha256": file_hash(Path(__file__))}
    metadata["evaluation_status_counts"] = ledger.loc[
        ledger["date"].isin(metadata["evaluation_dates"]), "status"].value_counts().to_dict()
    package = Path(__file__).resolve().parents[1]
    metadata["code_manifest"] = [{"file": str(path.relative_to(package.parent)), "sha256": file_hash(path)}
                                 for path in sorted(package.rglob("*.py"))]
    metadata["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "pyarrow": version("pyarrow")}
    output.mkdir(parents=True, exist_ok=False)
    all_signals.to_parquet(output / "signals.parquet", index=False)
    ledger.to_csv(output / "opportunities.csv", index=False)
    summary.to_csv(output / "regime_comparison.csv", index=False)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    audit_mask = (~ledger["status"].isin(["closed", "warmup"]) |
                  ledger.get("exit_delay_seconds", pd.Series(0, index=ledger.index)).gt(0))
    ledger.loc[audit_mask].to_csv(output / "execution_audit.csv", index=False)
    (output / "study.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output, metadata, summary)
    return metadata


def write_report(output: Path, metadata: Dict, summary: pd.DataFrame):
    cfg = metadata["settings"]
    lines = ["# QQQ 波动信号与固定合约跨式组合：历史验证", "",
             "阈值仅使用前 {} 个交易日相同半小时段的波动样本。前 {} 天用于初始化，随后 {} 天用于历史滚动验证。".format(
                 cfg["calibration_days"], len(metadata["calibration_dates"]), len(metadata["evaluation_dates"])), "",
             "这些日期已参与此前探索，不是新的独立留出样本。", "",
             "每 {} 秒评估一次；延迟 {} 秒，买入信号时选定的一张看涨与一张看跌；持有 {} 秒，再等待最多 {} 秒的可用双腿报价退出。同一时刻最多持有一组跨式组合。".format(
                 cfg["signal_seconds"], cfg["latency_seconds"], cfg["holding_seconds"], cfg["max_exit_delay_seconds"]), "",
             "按 ask 买、bid 卖，每次成交每张合约手续费 ${:.2f}，买入价格上调/卖出价格下调 {} bps，合约乘数 {}。报价新鲜度上限 {} 秒，双边报价量均至少一张。".format(
                 cfg["commission_per_contract"], cfg["slippage_bps"], cfg["multiplier"], cfg["max_quote_age_seconds"]), "",
             "| 波动状态 | 候选信号 | 完成退出 | 无入场报价 | 未找到合约 | 无法退出 | 延迟退出 | 中价毛损益/组 | 价差成本/组 | 滑点/组 | 手续费/组 | 净损益/组 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def money(value):
        return "—" if value is None or pd.isna(value) else "${:.3f}".format(value)
    for row in summary.to_dict("records"):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["regime"], row["signals"], row["closed"], row["entry_unavailable"], row["missing_contract"],
            row["unclosed"], row["delayed_exits"], *[money(row["mean_" + field]) for field in
            ("mid_pnl", "spread_cost", "slippage_cost", "commission", "net_pnl")]))
    lines += ["", "表中每组为一张看涨加一张看跌，平均损益仅针对已退出组合。无法退出指在规定等待窗口内未能退出。所有状态均保留在 opportunities.csv；缺失和延迟详情在 execution_audit.csv。",
              "无法退出的组合继续占用当日仓位，不记作零收益。regime_comparison.csv 同时给出其入场资金和按零回收计价的压力损益；压力损益不是实际到期结算值。", "",
              "## 所有已分类候选的状态", "",
              "| 波动状态 | 候选 | 已退出 | 入场报价不可用 | 未找到合约 | 仓位占用 | 临近收盘 | 超时未退出 | 未退出入场资金 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["regime"], row["signals"], row["closed"], row["entry_unavailable"], row["missing_contract"],
            row["busy"], row["session_end"], row["unclosed"], money(row["unclosed_debit"])))
    lines += ["", "初始化期和正股窗口无效的候选另列于 study.json 的 status_counts，不会用于成交后收益统计。", "",
              "## 按交易日配对的高低组差异", ""]
    for metric, comparison in metadata["paired_daily_comparisons"].items():
        low, high = comparison["daily_bootstrap_ci95"]
        lines.append("- {}：{} 个配对日期，高减低均值 {:.6f}；按日重抽样 95% 区间 [{:.6f}, {:.6f}]，{} 天为正。".format(
            metric, comparison["paired_days"], comparison["high_minus_low"], low, high, comparison["positive_days"]))
    lines += ["", "## 方法与边界", "",
              "- 从完整一秒正股网格重新计算过去 30 秒的波动；窗口中的过期、单边或无效报价会使该信号无效。校准完全不使用期权报价或未来能否退出。",
              "- 每日阈值由之前交易日决定，当天不更新；低于历史 25% 分位为 low，高于 75% 为 high，其他为 middle。阈值相同时不会产生高低组。",
              "- 合约来自信号时已经观测到的同到期日、同执行价配对深度记录；选择最接近当时 QQQ 中价的执行价，距离最多 ${}。之后按完整合约代码查询 normalized 报价，ATM 变化不会换约。".format(cfg["max_atm_distance"]),
              "- 无效的最新期权深度更新会使报价不可用；不会回退到更早的有效报价。入场只检查指定延迟时刻，退出从截止时刻按一秒检查，使用当时已收到的报价。",
              "- 低价过滤仅在入场应用，退出价格变低不会导致样本被删除。未入场、仓位占用、临近收盘和无法退出都有独立状态。",
              "- 两腿同时按可见报价成交是模拟假设，没有盘口抢单、部分成交或腿间风险模型。固定 09:30–16:00，未实现半日交易日历。",
              "- 校准数据也曾用于探索；这次结果只用于诊断，最终验证需固定本协议并在新增、未看过的日期上运行。",
              "- study.json 保存完整参数、输入内容校验值、代码校验值及依赖版本。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    for name in ("calibration_days", "min_history_samples", "signal_seconds", "latency_seconds",
                 "holding_seconds", "max_exit_delay_seconds"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(StudySettings(), name))
    for name in ("max_quote_age_seconds", "commission_per_contract", "slippage_bps"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(StudySettings(), name))
    args = vars(parser.parse_args(argv))
    data_root = args.pop("data_root").expanduser()
    output_root = args.pop("output_root") or resolve_path(load_config(), "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = output_root.expanduser() / ("causal_volatility_" + stamp)
    metadata = run_study(data_root, output, StudySettings(**args))
    print(json.dumps({"report_directory": output.name, "status_counts": metadata["status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
