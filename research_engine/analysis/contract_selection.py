"""Fixed underlying signals across observable option strike choices."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_engine.analysis.causal_volatility import StudySettings, read_books, file_hash
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.reversal_refinement import SIGNAL_COLUMNS
from research_engine.analysis.stock_strategy_search import HORIZONS, aggregate, fresh_gate, pick

CHOICES = {"atm": 0., "itm1": 1., "itm2": 2., "otm1": -1.}


class StrikeBooks:
    def __init__(self, books, points):
        self.books, self.points = books, points

    def select_leg(self, when, spot, right, max_distance):
        atm = self.books.select_leg(when, spot, right, max_distance)
        if atm is None:
            return None
        target = atm[0] + (-self.points if right == "CALL" else self.points)
        known = [item for item in self.books.contracts if item[1] == right and item[3] <= when.value and abs(item[0] - target) < 1e-9]
        return (known[0][0], known[0][2]) if known else None

    def quote(self, symbol, when, max_age):
        return self.books.quote(symbol, when, max_age)


def run(data_root, baseline_root, output):
    meta = json.loads((baseline_root / "study.json").read_text())
    if meta["context"] != "morning" or meta["strict_entry"]:
        raise ValueError("requires the original morning signal study")
    for record in meta["source_manifest"]:
        if file_hash(data_root / record["file"]) != record["sha256"]:
            raise ValueError("source data changed")
    output.mkdir(parents=True, exist_ok=False)
    daily, ledgers = [], []
    for day in meta["discovery_dates"] + meta["check_dates"]:
        source = pd.read_parquet(baseline_root / "opportunities" / (day + ".parquet"))
        signals = source[source.factor.eq("reversal60") & source.horizon.eq(300)][SIGNAL_COLUMNS].copy()
        underlying_books = read_books(data_root / "normalized" / day / "events.parquet", day)
        for choice, points in CHOICES.items():
            books = StrikeBooks(underlying_books, points)
            signals["factor"] = choice
            for horizon in HORIZONS:
                settings = replace(StudySettings(), holding_seconds=horizon)
                ledger = simulate_single_day(signals, books, settings, entry_gate=fresh_gate(books))
                ledger["horizon"] = horizon
                ledger["strike_offset_points"] = points
                ledgers.append(ledger)
                metrics = execution_metrics(ledger)
                metrics["entry_rejected"] = int(ledger.status.eq("entry_rejected").sum())
                daily.append(dict(date=day, factor=choice, horizon=horizon, **metrics))
        print("Completed " + day, flush=True)
    daily = pd.DataFrame(daily)
    ledger = pd.concat(ledgers, ignore_index=True)
    discovery = aggregate(daily, meta["discovery_dates"])
    selected = pick(discovery)
    checking = aggregate(daily, meta["check_dates"]).merge(selected[["factor", "horizon"]], on=["factor", "horizon"], how="inner")
    all_dates = aggregate(daily, meta["discovery_dates"] + meta["check_dates"])
    exposure = []
    for (choice, horizon), group in ledger.groupby(["factor", "horizon"]):
        complete = group[group.status.eq("closed")]
        paid = group[group.status.isin(["closed", "unclosed"])]
        exposure.append(dict(factor=choice, horizon=horizon, closed=len(complete),
                             mean_premium_return=complete.net_return.mean() if len(complete) else None,
                             mean_entry_debit=paid.entry_debit.mean() if len(paid) else None,
                             maximum_entry_debit=paid.entry_debit.max() if len(paid) else None))
    for name, values in [("daily", daily), ("discovery", discovery), ("selected", selected), ("finalists_check", checking),
                          ("all_dates", all_dates), ("premium_exposure", pd.DataFrame(exposure))]:
        values.to_csv(output / (name + ".csv"), index=False)
    ledger.to_parquet(output / "opportunities.parquet", index=False)
    package = Path(__file__).resolve().parents[2]
    protocol = package / "research/contract_selection_protocol.md"
    (output / "protocol.md").write_text(protocol.read_text(), encoding="utf-8")
    metadata = dict(study="fixed_stock_reversal_contract_choice", choices=CHOICES, horizons=HORIZONS,
                    settings=asdict(StudySettings()), discovery_dates=meta["discovery_dates"], check_dates=meta["check_dates"],
                    source_manifest=meta["source_manifest"], baseline_study=baseline_root.name,
                    baseline_metadata_sha256=file_hash(baseline_root / "study.json"), protocol_sha256=file_hash(protocol),
                    code_manifest=[dict(file=str(p.relative_to(package)), sha256=file_hash(p)) for p in sorted((package / "research_engine").rglob("*.py"))])
    (output / "study.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Discovery:"); print(discovery.to_string(index=False))
    print("Fixed-rule check:"); print(checking.to_string(index=False))
    print("Premium exposure:"); print(pd.DataFrame(exposure).to_string(index=False))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path("results/research") / ("contract_selection_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.data_root, args.baseline_root, output)


if __name__ == "__main__":
    main()
