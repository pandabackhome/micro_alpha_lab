"""Continue frozen paper research when complete new QQQ recordings arrive."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.adaptive_exit_search import underlying_exit_scheduler
from research_engine.analysis.active_settings import active_settings
from research_engine.analysis.causal_volatility import BOOK_COLUMNS, StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import daily_ci, execution_metrics, simulate_single_day
from research_engine.analysis.event_reversal_search import crossing_signals
from research_engine.analysis.frozen_dataset import research_inputs
from research_engine.analysis.reversal_execution_audit import audit_raw_fills
from research_engine.analysis.search_artifact_audit import validate_ledger
from research_engine.analysis.stock_strategy_search import COLUMNS, calibrate, fresh_gate, signals_for, stock_observations
from research_engine.cli import build_day, recording_index
from research_engine.config import load_config, resolve_path


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,default=str,ensure_ascii=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def fingerprint(files):
    values=[]
    for path in files:
        sidecar=path.with_name(path.name.replace(".jsonl.gz",".jsonl.source.sha256"))
        if not sidecar.exists() or path.stat().st_size==0:
            return None
        for item in (path,sidecar):
            stat=item.stat()
            values.append((item.name,stat.st_size,stat.st_mtime_ns))
    return tuple(values)


def stable_recording(day, files, seen, now, stable_seconds):
    state=fingerprint(files)
    if state is None:
        seen.pop(day,None)
        return False
    if day not in seen or seen[day][0]!=state:
        seen[day]=(state,now)
        return False
    return now-seen[day][1]>=stable_seconds


def dataset_status(frame, raw_stock, day):
    valid=frame.qqq_depth_age_seconds.between(0,5)&frame.qqq_bid.gt(0)&frame.qqq_ask.ge(frame.qqq_bid)
    valid &= np.isfinite(frame.qqq_bid)&np.isfinite(frame.qqq_ask)
    stamps=pd.to_datetime(raw_stock.available_at,utc=True)
    regular_close=pd.Timestamp(day+" 16:00",tz="America/New_York").tz_convert("UTC")
    last=stamps[stamps.le(regular_close)].max()
    coverage=float(valid.mean())
    ready=coverage>=.95 and pd.notna(last) and last>=regular_close-pd.Timedelta(seconds=30)
    return dict(ready=bool(ready),fresh_stock_fraction=coverage,last_regular_stock_quote=str(last))


def evaluate_day(day,local_data,history_paths,settings,output):
    history=[]
    sources=[]
    for path in history_paths[-5:]:
        history.append(stock_observations(pd.read_parquet(path,columns=COLUMNS),settings))
        sources.append(dict(role="calibration",file=path.name,sha256=file_hash(path)))
    if len(history)!=5 or any(item.date.iloc[0]>=day for item in history):
        raise ValueError("five completed earlier dates required")
    feature=local_data/"features"/(day+".parquet")
    normalized=local_data/"normalized"/day/"events.parquet"
    frame=pd.read_parquet(feature,columns=COLUMNS)
    raw_stock=pd.read_parquet(normalized,columns=["available_at"],filters=[("kind","=","depth"),("symbol","=","QQQ.US")])
    quality=dataset_status(frame,raw_stock,day)
    if not quality["ready"]:
        return None,quality
    observed=stock_observations(frame,settings)
    if set(observed.date)!={day}:
        raise ValueError("recording date mismatch")
    books=read_books(normalized,day)
    raw=pd.read_parquet(normalized,columns=BOOK_COLUMNS,filters=[("kind","=","depth"),("symbol","in",list(books.books))])
    grid=signals_for(calibrate(observed,history,settings),"morning")
    grid=grid[grid.factor.eq("reversal60")]
    event=crossing_signals(calibrate(stock_observations(frame,settings,sampling_seconds=1),history,settings),mode="recovery")
    daily,ledgers=[],[]
    quote_checks=0
    for factor,signals,scheduler in [("clock_reversal",grid,None),
                                     ("clock_reversal_stop",grid,underlying_exit_scheduler(frame,None,.5)),
                                     ("recovery_reversal",event,None)]:
        ledger=simulate_single_day(signals.assign(factor=factor),books,settings,entry_gate=fresh_gate(books),exit_scheduler=scheduler)
        ledger["horizon"]=300
        validate_ledger(ledger,settings.multiplier)
        quote_checks+=audit_raw_fills(ledger,raw,settings)
        metrics=execution_metrics(ledger)
        metrics["entry_rejected"]=int(ledger.status.eq("entry_rejected").sum())
        daily.append(dict(date=day,factor=factor,horizon=300,**metrics))
        ledgers.append(ledger)
    output.mkdir(parents=True,exist_ok=False)
    daily=pd.DataFrame(daily)
    daily.to_csv(output/"daily.csv",index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(output/"opportunities.parquet",index=False)
    sources += [dict(role="evaluation_features",file="features/"+feature.name,sha256=file_hash(feature)),
                dict(role="evaluation_quotes",file="normalized/"+day+"/events.parquet",sha256=file_hash(normalized))]
    write_json(output/"study.json",dict(date=day,settings=asdict(settings),source_manifest=sources,quality=quality,
                                       raw_quotes_checked=quote_checks,history_dates=[item.date.iloc[0] for item in history],
                                       complete=True))
    (output/"report.md").write_text("\n".join(["# 冻结规则后续录制检验："+day,"",
        "新录制完成后回放，规则和费用沿用启动时冻结版本；未做参数选择，也未发送真实订单。","",
        "```",daily.to_string(index=False),"```",""]),encoding="utf-8")
    return daily,quality


def summarize_watch(root):
    reports=sorted((root/"days").glob("*/daily.csv"))
    if not reports:
        return []
    daily=pd.concat([pd.read_csv(path) for path in reports],ignore_index=True)
    daily.to_csv(root/"daily.csv",index=False)
    summaries=[]
    for factor,group in daily.groupby("factor"):
        low,high=daily_ci(group.mean_net_pnl)
        net=float(group.net_with_unclosed_zero_recovery.sum())
        without_best=net-float(group.net_with_unclosed_zero_recovery.max())
        days=len(group)
        candidate=(days>=10 and group.closed.sum()>=30 and group.net_with_unclosed_zero_recovery.gt(0).sum()>=np.ceil(.6*days)
                   and net>0 and without_best>0 and low is not None and low>0)
        summaries.append(dict(factor=factor,days=days,closed=int(group.closed.sum()),stress_net=net,without_best_day=without_best,
                              positive_days=int(group.net_with_unclosed_zero_recovery.gt(0).sum()),daily_ci_low=low,daily_ci_high=high,
                              provisional_candidate=bool(candidate)))
    pd.DataFrame(summaries).to_csv(root/"summary.csv",index=False)
    return summaries


def run(archive_data,recordings_root,root,poll_seconds=30,stable_seconds=60,once=False,fee=None):
    if poll_seconds<1 or stable_seconds<0:
        raise ValueError("invalid polling intervals")
    package=Path(__file__).resolve().parents[2]
    dataset=package/"research/complete_calendar_dataset.json"
    protocol=package/"research/forward_watch_protocol.md"
    paths,_=research_inputs(archive_data,package/"data",dataset)
    settings=active_settings(holding_seconds=300,**({"commission_per_contract":fee} if fee is not None else {}))
    fee=settings.commission_per_contract
    config=load_config()
    config["paths"]["recordings"]=str(recordings_root.resolve())
    local_data=resolve_path(config,"data")
    root.mkdir(parents=True,exist_ok=True)
    (root/"days").mkdir(exist_ok=True)
    freeze_path=root/"freeze.json"
    expected=dict(settings=asdict(settings),dataset_sha256=file_hash(dataset),protocol_sha256=file_hash(protocol),
                  cutoff=max(paths),rules=["clock_reversal","clock_reversal_stop","recovery_reversal"])
    if freeze_path.exists():
        freeze=json.loads(freeze_path.read_text())
        if any(freeze[key]!=value for key,value in expected.items()):
            raise ValueError("existing watch has a different frozen configuration; use a new watch directory")
    else:
        freeze=dict(expected,created_at=datetime.now(timezone.utc).isoformat(),
                    code_manifest=[dict(file=str(p.relative_to(package)),sha256=file_hash(p)) for p in sorted((package/"research_engine").rglob("*.py"))])
        write_json(freeze_path,freeze)
        (root/"protocol.md").write_bytes(protocol.read_bytes())
        (root/"dataset.json").write_bytes(dataset.read_bytes())
    completed={p.parent.name for p in (root/"days").glob("*/study.json") if json.loads(p.read_text()).get("complete")}
    for day in completed:
        paths[day]=("local",local_data/"features"/(day+".parquet"),local_data/"normalized"/day/"events.parquet")
    seen={}
    previous=None
    while True:
        status=dict(updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),state="waiting_for_new_recording",
                    last_completed_date=max(paths),completed_forward_dates=sorted(completed),fee_per_side=fee,
                    paper_only=True,poll_seconds=poll_seconds,stable_seconds=stable_seconds)
        for item in freeze["code_manifest"]:
            if file_hash(package/item["file"])!=item["sha256"]:
                status.update(state="stopped_code_changed",changed_file=item["file"])
                write_json(root/"status.json",status)
                print(json.dumps(status,ensure_ascii=False),flush=True)
                return
        candidates=recording_index(config,verbose=False)
        new_days=sorted(day for day in candidates if day>max(paths) and day not in completed)
        if new_days:
            day=new_days[0]
            status.update(state="waiting_for_stable_recording",next_date=day)
            if stable_recording(day,candidates[day],seen,time.monotonic(),stable_seconds):
                try:
                    raw_manifest=[dict(file=p.name,compressed_sha256=file_hash(p)) for p in candidates[day]]
                    status.update(state="building_and_evaluating",next_date=day)
                    write_json(root/"status.json",status)
                    build_day(config,day)
                    history_paths=[paths[key][1] for key in sorted(paths) if key<day][-5:]
                    daily,quality=evaluate_day(day,local_data,history_paths,settings,root/"days"/day)
                    if daily is None:
                        status.update(state="waiting_for_complete_session",quality=quality)
                    else:
                        write_json(root/"days"/day/"recordings.json",raw_manifest)
                        completed.add(day)
                        paths[day]=("local",local_data/"features"/(day+".parquet"),local_data/"normalized"/day/"events.parquet")
                        status.update(state="evaluated_new_date",last_completed_date=day,completed_forward_dates=sorted(completed),
                                      summary=summarize_watch(root))
                except Exception as exc:
                    message=str(exc)
                    for path,label in [(package,"{project}"),(archive_data.parent,"{archive}"),(recordings_root,"{recordings}")]:
                        message=message.replace(str(path.resolve()),label)
                    status.update(state="retry_after_error",error_type=type(exc).__name__,error=message)
        write_json(root/"status.json",status)
        signature=(status["state"],status.get("next_date"),status["last_completed_date"],status.get("error"))
        if signature!=previous:
            print(json.dumps(status,ensure_ascii=False),flush=True)
            previous=signature
        if once:
            return
        time.sleep(poll_seconds)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data",type=Path,required=True)
    parser.add_argument("--recordings-root",type=Path,required=True)
    parser.add_argument("--watch-root",type=Path,default=Path("results/research/forward_watch"))
    parser.add_argument("--poll-seconds",type=float,default=30)
    parser.add_argument("--stable-seconds",type=float,default=60)
    parser.add_argument("--fee-per-side",type=float,help="override current configured fee")
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args(argv)
    run(args.archive_data,args.recordings_root,args.watch_root,args.poll_seconds,args.stable_seconds,args.once,args.fee_per_side)


if __name__=="__main__":
    main()
