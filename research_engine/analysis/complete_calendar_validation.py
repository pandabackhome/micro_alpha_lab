"""Frozen rules with an explicitly pinned, newly completed trading calendar."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_engine.analysis.adaptive_exit_search import underlying_exit_scheduler
from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.event_reversal_search import crossing_signals
from research_engine.analysis.frozen_dataset import research_inputs
from research_engine.analysis.stock_strategy_search import COLUMNS, aggregate, calibrate, fresh_gate, signals_for, stock_observations


def run(archive_data, local_data, manifest_path, output):
    paths, dataset = research_inputs(archive_data, local_data, manifest_path)
    settings=replace(StudySettings(),holding_seconds=300)
    output.mkdir(parents=True,exist_ok=False)
    history,daily,ledgers,calibration=[],[],[],[]
    for day in sorted(paths):
        origin,feature_path,normalized_path=paths[day]
        frame=pd.read_parquet(feature_path,columns=COLUMNS)
        observed=stock_observations(frame,settings)
        if set(observed.date)!={day}:
            raise ValueError("feature date mismatch")
        if len(history)>=5:
            books=read_books(normalized_path,day)
            calibration.append(dict(date=day,history_dates="|".join(item.date.iloc[0] for item in history[-5:])))
            grid=signals_for(calibrate(observed,history,settings),"morning")
            grid=grid[grid.factor.eq("reversal60")]
            event=crossing_signals(calibrate(stock_observations(frame,settings,sampling_seconds=1),history,settings),mode="recovery")
            for name,signals,scheduler in [
                ("clock_reversal",grid,None),
                ("clock_reversal_stop",grid,underlying_exit_scheduler(frame,None,.5)),
                ("recovery_reversal",event,None)]:
                ledger=simulate_single_day(signals.assign(factor=name),books,settings,entry_gate=fresh_gate(books),exit_scheduler=scheduler)
                ledger["horizon"]=300
                metrics=execution_metrics(ledger)
                metrics["entry_rejected"]=int(ledger.status.eq("entry_rejected").sum())
                daily.append(dict(date=day,factor=name,horizon=300,**metrics))
                ledgers.append(ledger)
            print("Replayed complete calendar "+day,flush=True)
        history.append(observed)
    daily=pd.DataFrame(daily)
    groups=[("original_discovery",dataset["discovery_dates"]),("original_check",dataset["check_dates"]),
            ("added_reused",dataset["added_dates"]),("backfilled",dataset["backfilled_dates"])]
    summary=pd.concat([aggregate(daily,dates).assign(part=name) for name,dates in groups],ignore_index=True)
    daily.to_csv(output/"daily.csv",index=False)
    summary.to_csv(output/"summary.csv",index=False)
    pd.DataFrame(calibration).to_csv(output/"calibration.csv",index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(output/"opportunities.parquet",index=False)
    package=Path(__file__).resolve().parents[2]
    protocol=package/"research/complete_calendar_protocol.md"
    (output/"protocol.md").write_bytes(protocol.read_bytes())
    (output/"dataset.json").write_bytes(manifest_path.read_bytes())
    metadata=dict(study="frozen_rules_complete_calendar",settings=asdict(settings),dates=sorted(paths),groups=dict(groups),
                  source_manifest=dataset["sources"],dataset_manifest_sha256=file_hash(manifest_path),protocol_sha256=file_hash(protocol),
                  code_manifest=[dict(file=str(p.relative_to(package)),sha256=file_hash(p)) for p in sorted((package/"research_engine").rglob("*.py"))])
    (output/"study.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    (output/"report.md").write_text("\n".join([
        "# 补齐日期后的固定规则重放","","没有改变规则，但补齐9月9、10日后，后续历史校准窗口发生变化。",
        "补齐日期不是严格时间外验证；其他日期已用于探索。旧冻结结果保持原样，不用本轮覆盖。","",
        "## 分组结果","","```",summary.to_string(index=False),"```","",
        "## 每日历史窗口","","```",pd.DataFrame(calibration).to_string(index=False),"```",""]),encoding="utf-8")
    print(summary.to_string(index=False))
    print("Report: "+output.name)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data",type=Path,required=True)
    parser.add_argument("--local-data",type=Path,default=Path("data"))
    parser.add_argument("--dataset",type=Path,default=Path("research/complete_calendar_dataset.json"))
    args=parser.parse_args(argv)
    output=Path("results/research")/("complete_calendar_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data,args.local_data,args.dataset,output)


if __name__=="__main__":
    main()
