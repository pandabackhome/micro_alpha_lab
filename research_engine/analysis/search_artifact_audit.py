"""Validate saved search provenance, trade cash flows and position accounting."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import file_hash
from research_engine.analysis.causal_volatility import BOOK_COLUMNS, StudySettings
from research_engine.analysis.reversal_execution_audit import audit_raw_fills

PREFIXES = ("stock_strategy_", "stock_prediction_", "reversal_refinement_", "contract_selection_",
            "option_target_", "reversal_execution_", "adaptive_exit_", "reversal_phase_", "event_reversal_",
            "new_dates_validation_", "vertical_spread_", "position_capacity_", "long_context_",
            "complete_calendar_", "rich_stock_model_")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_ledger(ledger, multiplier):
    paid = ledger[ledger.status.isin(["closed", "unclosed"])]
    closed = ledger[ledger.status.eq("closed")]
    unresolved = ledger[ledger.status.eq("unclosed")]
    if len(closed):
        cash = (closed.exit_fill - closed.entry_fill) * multiplier - closed.commission
        require(np.allclose(cash, closed.net_pnl, rtol=0, atol=1e-8), "cash flow identity failed")
        cost = closed.mid_pnl - closed.spread_cost - closed.slippage_cost - closed.commission
        require(np.allclose(cost, closed.net_pnl, rtol=0, atol=1e-8), "cost decomposition failed")
        require(np.allclose(closed.net_return, closed.net_pnl / closed.entry_debit), "return identity failed")
        require(closed.exit_time.ge(closed.exit_due).all(), "exit before known deadline")
        require(closed.exit_time.le(closed.session_close).all(), "exit after session")
    if len(paid):
        require(paid.entry_time.gt(paid.timestamp).all(), "entry before routing delay")
        require(np.allclose(paid.entry_debit, paid.entry_fill * multiplier + paid.entry_fee), "entry debit identity failed")
        if "checked_entry_age" in paid:
            require(paid.checked_entry_age.dropna().between(0, 1).all(), "strict entry age failed")
        if "maximum_exit_due" in paid:
            adaptive = paid[paid.maximum_exit_due.notna()]
            require(adaptive.exit_due.le(adaptive.maximum_exit_due).all(), "adaptive exit exceeds deadline")
            triggered = adaptive[adaptive.exit_trigger_time.notna()]
            require(triggered.exit_due.gt(triggered.exit_trigger_time).all(), "exit lacks reaction delay")
    if len(unresolved) and "net_pnl" in unresolved:
        require(unresolved.net_pnl.isna().all(), "unclosed trade has invented realized PnL")
    for _, group in paid.groupby(["date", "factor", "horizon"]):
        group = group.sort_values("timestamp")
        if "capacity" in group:
            for index, row in enumerate(group.itertuples()):
                previous = group.iloc[:index]
                end = previous.exit_time.where(previous.status.eq("closed"), previous.session_close)
                require(int(end.gt(row.entry_time).sum()) < row.capacity, "position capacity exceeded")
            require(group.cash_after_entry.ge(-1e-8).all(), "negative available cash")
            require(np.allclose(group.cash_before_entry-group.cash_after_entry,group.entry_debit), "cash entry accounting mismatch")
            continue
        if len(group) > 1:
            previous_end = group.exit_time.where(group.status.eq("closed"), group.session_close).iloc[:-1]
            require((group.timestamp.iloc[1:].reset_index(drop=True) >= previous_end.reset_index(drop=True)).all(),
                    "overlapping positions in one strategy")
    for name in ("history_last_date", "history_first_date"):
        if name in ledger:
            history = ledger[ledger[name].notna()]
            require((history[name] < history.date).all(), "future date used as history")
    return len(closed), len(unresolved)


def validate_spreads(ledger, settings):
    paid = ledger[ledger.status.isin(["closed","unclosed"])]
    closed = ledger[ledger.status.eq("closed")]
    unresolved = ledger[ledger.status.eq("unclosed")]
    entry = (paid.entry_short_fill-paid.entry_long_fill)*settings.multiplier-2*settings.commission_per_contract
    require(np.allclose(entry,paid.entry_credit),"spread entry cash mismatch")
    require(np.allclose(paid.maximum_loss_model,paid.width*settings.multiplier-entry+2*settings.commission_per_contract),"spread risk identity failed")
    require(paid.short_symbol.ne(paid.long_symbol).all(),"missing distinct protection leg")
    require(np.allclose((paid.short_strike-paid.long_strike).abs(),paid.width),"spread width mismatch")
    debit=(closed.exit_short_fill-closed.exit_long_fill)*settings.multiplier+2*settings.commission_per_contract
    require(np.allclose(debit,closed.exit_debit),"spread exit cash mismatch")
    require(np.allclose(closed.entry_credit-debit,closed.net_pnl),"spread net cash mismatch")
    for _, group in paid.groupby(["date","factor","horizon"]):
        group=group.sort_values("timestamp")
        if len(group)>1:
            end=group.exit_time.where(group.status.eq("closed"),group.session_close).iloc[:-1].reset_index(drop=True)
            require((group.timestamp.iloc[1:].reset_index(drop=True)>=end).all(),"overlapping spreads")
    return len(closed),len(unresolved)


def audit_spread_quotes(ledger, raw):
    source=raw.copy()
    source["available_at"]=pd.to_datetime(source.available_at,utc=True).dt.as_unit("ns")
    groups={symbol:group.sort_values("available_at",kind="stable") for symbol,group in source.groupby("symbol")}
    checked=0
    for row in ledger[ledger.status.isin(["closed","unclosed"])].itertuples():
        for leg,symbol,strike in [("short",row.short_symbol,row.short_strike),("long",row.long_symbol,row.long_strike)]:
            group=groups[symbol]
            require(group.available_at.min()<=row.timestamp,"protection not known at signal")
            sides=[("entry",row.entry_time)]
            if row.status=="closed":
                sides.append(("exit",row.exit_time))
            for side,when in sides:
                eligible=group[group.available_at.le(when)]
                require(not eligible.empty,"missing raw spread quote")
                quote=eligible.iloc[-1]
                age=(when-quote.available_at).total_seconds()
                require(0<=age<=1 and quote.best_bid>0 and quote.best_ask>=quote.best_bid and
                        quote.best_bid_size>=1 and quote.best_ask_size>=1,"invalid raw spread quote")
                require(quote.option_strike==strike and quote.option_right==row.right,"spread contract mismatch")
                price_side="bid" if (side,leg) in [("entry","short"),("exit","long")] else "ask"
                price=getattr(quote,"best_"+price_side)
                actual=getattr(row,side+"_"+leg+"_"+price_side)
                require(np.isclose(price,actual,rtol=0,atol=1e-9),"raw spread price mismatch")
                require(np.isclose(age,getattr(row,side+"_"+leg+"_age"),rtol=0,atol=1e-6),"raw spread age mismatch")
                if side=="entry":
                    require(quote.available_at>row.timestamp,"spread entry quote predates signal")
                checked+=1
    return checked


def run(data_root, results_root, local_data=Path("data"), study_prefixes=None):
    package = Path(__file__).resolve().parents[2]
    checked_hashes, studies = {}, []
    def cached_hash(path):
        key = str(path)
        if key not in checked_hashes:
            checked_hashes[key] = file_hash(path)
        return checked_hashes[key]
    for root in sorted(results_root.iterdir()):
        if not root.name.startswith(PREFIXES) or not (root / "study.json").exists():
            continue
        if study_prefixes and not root.name.startswith(tuple(study_prefixes)):
            continue
        meta = json.loads((root / "study.json").read_text())
        archived = 0
        for entry in meta["code_manifest"]:
            source = package / entry["file"]
            if cached_hash(source) != entry["sha256"]:
                snapshot = results_root / "source_snapshots" / entry["sha256"] / entry["file"]
                require(snapshot.exists() and cached_hash(snapshot) == entry["sha256"], "missing matching code version")
                archived += 1
        normalized_sources={}
        for entry in meta["source_manifest"]:
            source_root=local_data if entry.get("origin")=="local" else data_root
            source_path=source_root/entry["file"]
            require(cached_hash(source_path) == entry["sha256"], "source data hash changed")
            if entry["file"].startswith("normalized/"):
                normalized_sources[Path(entry["file"]).parts[1]]=source_path
        require(cached_hash(root / "protocol.md") == meta["protocol_sha256"], "protocol snapshot changed")
        paths = [root / "opportunities.parquet"] if (root / "opportunities.parquet").exists() else sorted((root / "opportunities").glob("*.parquet"))
        require(bool(paths), "no opportunity ledgers")
        parts = [pd.read_parquet(path) for path in paths]
        ledger = pd.concat(parts, ignore_index=True)
        spread=meta["study"]=="stock_events_protected_credit_spreads"
        settings=StudySettings(**meta["settings"])
        closed, unclosed = validate_spreads(ledger,settings) if spread else validate_ledger(ledger,settings.multiplier)
        raw_checked=0
        if root.name.startswith(("new_dates_validation_","vertical_spread_","position_capacity_","long_context_",
                                 "complete_calendar_","rich_stock_model_")):
            for day,group in ledger.groupby("date"):
                symbols=list(set(group.short_symbol.dropna())|set(group.long_symbol.dropna())) if spread else list(group.symbol.dropna().unique())
                raw=pd.read_parquet(normalized_sources[day],columns=BOOK_COLUMNS,filters=[("kind","=","depth"),("symbol","in",symbols)])
                raw_checked+=audit_spread_quotes(group,raw) if spread else audit_raw_fills(group,raw,settings)
        daily = pd.read_csv(root / "daily.csv")
        for row in daily.itertuples():
            group = ledger[ledger.date.eq(row.date) & ledger.factor.eq(row.factor) & ledger.horizon.eq(row.horizon)]
            done = group[group.status.eq("closed")]
            unresolved = group[group.status.eq("unclosed")]
            net = done.net_pnl.sum() if len(done) else 0.
            debit = (unresolved.maximum_loss_model if spread else unresolved.entry_debit).sum() if len(unresolved) else 0.
            require(len(done) == row.closed and len(unresolved) == row.unclosed, "daily counts mismatch")
            require(np.isclose(net, row.realized_net_pnl), "daily realized PnL mismatch")
            require(np.isclose(net - debit, row.net_with_unclosed_zero_recovery), "daily pressure PnL mismatch")
            if "end_cash" in daily:
                require(np.isclose(row.end_cash-row.initial_capital,net-debit),"daily cash conservation failed")
        training = root / "training_audit.csv"
        if training.exists():
            audit = pd.read_csv(training)
            require((audit.history_last_date < audit.date).all(), "training date leakage")
            decisions = pd.read_parquet(root / "signals.parquet")
            require(not any(name.startswith("target_") for name in decisions), "training labels in decision table")
        studies.append(dict(study=root.name, rows=len(ledger), closed=closed, unclosed=unclosed,
                            code_files=len(meta["code_manifest"]), archived_code_files=archived,
                            source_files=len(meta["source_manifest"]),raw_quotes_checked=raw_checked, passed=True))
        print("Passed " + root.name, flush=True)
    output = results_root / ("search_validation_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(studies).to_csv(output / "studies.csv", index=False)
    report = dict(studies=len(studies), ledger_rows=sum(row["rows"] for row in studies),
                  closed_records=sum(row["closed"] for row in studies),
                  unclosed_records=sum(row["unclosed"] for row in studies), all_passed=True,
                  raw_quotes_checked=sum(row["raw_quotes_checked"] for row in studies),
                  note="Records across strategies and repeated audits are dependent; counts are not independent trades.",
                  audit_code_sha256=file_hash(Path(__file__)),
                  study_manifest=[dict(study=row["study"], sha256=file_hash(results_root / row["study"] / "study.json")) for row in studies])
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "study_manifest"}))
    print("Report: " + output.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/research"))
    parser.add_argument("--local-data", type=Path, default=Path("data"))
    parser.add_argument("--study-prefix",action="append",help="audit only named result prefixes; repeat for several")
    args = parser.parse_args(argv)
    run(args.data_root, args.results_root, args.local_data,args.study_prefix)


if __name__ == "__main__":
    main()
