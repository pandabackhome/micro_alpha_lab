"""Causal follow-up of the morning reversal candidate; no horizon retuning."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import StudySettings, read_books, file_hash
from research_engine.analysis.directional_factors import daily_ci, execution_metrics, future_outcomes, simulate_single_day
from research_engine.analysis.stock_strategy_search import COLUMNS, aggregate, fresh_gate, pick, stock_observations

VARIANTS = ("baseline", "range_only", "turn_confirmed", "range_and_turn")
SIGNAL_COLUMNS = ["timestamp", "date", "half_hour", "spot", "past_vol", "session_close", "scheduled",
                  "regime", "direction", "signal_status", "history_last_date", "threshold_r60", "r60", "spread"]


def efficiency_observations(frame, settings):
    observed = stock_observations(frame, settings)
    good = (frame.qqq_depth_age_seconds.between(0, 5) & frame.qqq_bid.gt(0) & frame.qqq_ask.ge(frame.qqq_bid) &
            np.isfinite(frame.qqq_bid) & np.isfinite(frame.qqq_ask))
    mid = ((frame.qqq_bid + frame.qqq_ask) / 2).where(good)
    denominator = mid.diff().abs().rolling(300, min_periods=300).sum()
    efficiency = (mid - mid.shift(300)).abs() / denominator.where(denominator.gt(0))
    values = pd.DataFrame({"timestamp": pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns"), "efficiency": efficiency})
    return observed.merge(values, on="timestamp", validate="one_to_one")


def efficiency_thresholds(current, history):
    result = current[["timestamp", "efficiency"]].copy()
    result["efficiency_median"] = np.nan
    if len(history) < 5:
        return result
    past = pd.concat(history[-5:], ignore_index=True)
    if past.date.max() >= current.date.min():
        raise ValueError("history must contain earlier dates")
    for bucket, idx in current.groupby("half_hour").groups.items():
        values = past.loc[past.half_hour.eq(bucket) & past.valid300, "efficiency"].dropna()
        if len(values) >= 100:
            result.loc[idx, "efficiency_median"] = values.median()
    return result


def refine(signals, frame, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown variant")
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).as_unit("ns")
    rows = []
    for signal in signals.to_dict("records"):
        row = dict(signal, factor=variant, origin_timestamp=signal["timestamp"], origin_spot=signal["spot"])
        rows.append(row)
        if signal["signal_status"] != "signal":
            continue
        if variant in ("range_only", "range_and_turn"):
            if not np.isfinite([signal["efficiency"], signal["efficiency_median"]]).all():
                row.update(signal_status="efficiency_unavailable", direction=0)
                continue
            if signal["efficiency"] >= signal["efficiency_median"]:
                row.update(signal_status="range_filtered", direction=0)
                continue
        if variant not in ("turn_confirmed", "range_and_turn"):
            continue
        confirmed = False
        for seconds in range(1, 31):
            when = signal["timestamp"] + pd.Timedelta(seconds=seconds)
            before = when - pd.Timedelta(seconds=5)
            local = when.tz_convert("America/New_York")
            if when not in frame.index or before not in frame.index or local.hour * 60 + local.minute >= 690:
                break
            now, old = frame.loc[when], frame.loc[before]
            if not all(0 <= item.qqq_depth_age_seconds <= 1 and np.isfinite([item.qqq_bid, item.qqq_ask]).all() and 0 < item.qqq_bid <= item.qqq_ask for item in (now, old)):
                continue
            mid, old_mid = (now.qqq_bid + now.qqq_ask) / 2, (old.qqq_bid + old.qqq_ask) / 2
            direction = signal["direction"]
            if direction * (mid - old_mid) > 0 and direction * (mid - signal["spot"]) >= signal["spread"]:
                row.update(timestamp=when, spot=mid, confirmation_seconds=seconds,
                           confirmation_change=mid - signal["spot"])
                confirmed = True
                break
        if not confirmed:
            row.update(signal_status="confirmation_timeout", direction=0)
    return pd.DataFrame(rows)


def run(data_root, baseline_root, output):
    settings = replace(StudySettings(), holding_seconds=300)
    meta = json.loads((baseline_root / "study.json").read_text())
    if meta["context"] != "morning" or meta["strict_entry"]:
        raise ValueError("requires the original morning search as baseline")
    for record in meta["source_manifest"]:
        if file_hash(data_root / record["file"]) != record["sha256"]:
            raise ValueError("source data changed: " + record["file"])
    output.mkdir(parents=True, exist_ok=False)
    history, ledgers, daily, label_rows, signals_saved = [], [], [], [], []
    for day in meta["dates"]:
        frame = pd.read_parquet(data_root / "features" / (day + ".parquet"), columns=COLUMNS)
        current = efficiency_observations(frame, settings)
        thresholds = efficiency_thresholds(current, history)
        history.append(current)
        if day not in meta["discovery_dates"] + meta["check_dates"]:
            continue
        baseline = pd.read_parquet(baseline_root / "opportunities" / (day + ".parquet"))
        signals = baseline[baseline.factor.eq("reversal60") & baseline.horizon.eq(300)][SIGNAL_COLUMNS].copy()
        signals = signals.merge(thresholds, on="timestamp", validate="one_to_one")
        for horizon in (60, 180, 300):
            labelled = signals[signals.signal_status.eq("signal")].merge(future_outcomes(frame, replace(settings, holding_seconds=horizon)), on="timestamp", validate="one_to_one")
            labelled["horizon"] = horizon
            label_rows.append(labelled)
        books = read_books(data_root / "normalized" / day / "events.parquet", day)
        for variant in VARIANTS:
            prepared = refine(signals, frame, variant)
            signals_saved.append(prepared)
            ledger = simulate_single_day(prepared, books, settings, entry_gate=fresh_gate(books))
            ledger["horizon"] = 300
            ledgers.append(ledger)
            metrics = execution_metrics(ledger)
            metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
            daily.append(dict(date=day, factor=variant, horizon=300, **metrics))
        print("Completed " + day, flush=True)
    daily = pd.DataFrame(daily)
    ledger = pd.concat(ledgers, ignore_index=True)
    labels = pd.concat(label_rows, ignore_index=True)
    discovery = aggregate(daily, meta["discovery_dates"])
    selected = pick(discovery)
    checking = aggregate(daily, meta["check_dates"]).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    overall = aggregate(daily, meta["discovery_dates"] + meta["check_dates"])
    direction_daily = []
    for (day, horizon), group in labels.groupby(["date", "horizon"]):
        good = group[group.delayed_return_bps.notna()]
        signed = good.direction * good.delayed_return_bps
        direction_daily.append(dict(date=day, horizon=horizon, signals=len(group), labelled=len(good),
                                    mean_signed_bps=signed.mean(), hit_rate=signed.gt(0).mean()))
    direction_daily = pd.DataFrame(direction_daily)
    direction = []
    for name, dates in [("discovery", meta["discovery_dates"]), ("check", meta["check_dates"])]:
        for horizon, group in direction_daily[direction_daily.date.isin(dates)].groupby("horizon"):
            lo, hi = daily_ci(group.mean_signed_bps)
            direction.append(dict(part=name, horizon=horizon, signals=int(group.signals.sum()), labelled=int(group.labelled.sum()),
                                  daily_mean_signed_bps=group.mean_signed_bps.mean(), ci_low=lo, ci_high=hi,
                                  positive_days=int(group.mean_signed_bps.gt(0).sum())))
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", checking),
                          ("all_dates", overall), ("stock_direction_daily", direction_daily), ("stock_direction", pd.DataFrame(direction))]:
        values.to_csv(output / (name + ".csv"), index=False)
    ledger.to_parquet(output / "opportunities.parquet", index=False)
    pd.concat(signals_saved).to_parquet(output / "signals.parquet", index=False)
    labels.to_parquet(output / "stock_evaluation.parquet", index=False)
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/reversal_refinement_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    metadata = dict(study="morning_reversal_refinement", variants=VARIANTS, discovery_dates=meta["discovery_dates"],
                    check_dates=meta["check_dates"], baseline_study=baseline_root.name,
                    baseline_metadata_sha256=file_hash(baseline_root / "study.json"), source_manifest=meta["source_manifest"],
                    protocol_sha256=file_hash(protocol), settings=asdict(settings),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Discovery:"); print(discovery.to_string(index=False))
    print("Fixed-rule check:"); print(checking.to_string(index=False))
    print("Underlying direction:"); print(pd.DataFrame(direction).to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("reversal_refinement_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, args.baseline_root, output)


if __name__ == "__main__":
    main()
