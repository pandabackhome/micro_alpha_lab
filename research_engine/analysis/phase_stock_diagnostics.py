"""Evaluation-only reversal diagnosis across all four minute phases and dates."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import StudySettings, file_hash
from research_engine.analysis.directional_factors import daily_ci, future_outcomes
from research_engine.analysis.frozen_dataset import research_inputs
from research_engine.analysis.stock_strategy_search import COLUMNS, calibrate, stock_observations


def run(archive_data,local_data,dataset_path,output):
    paths,dataset=research_inputs(archive_data,local_data,dataset_path)
    settings=StudySettings()
    output.mkdir(parents=True,exist_ok=False)
    history,daily,panels=[],[],[]
    for day in sorted(paths):
        _,feature_path,_=paths[day]
        frame=pd.read_parquet(feature_path,columns=COLUMNS)
        ordinary=stock_observations(frame,settings)
        if len(history)>=5:
            current=calibrate(stock_observations(frame,settings,sampling_seconds=15),history,settings)
            eligible=current.valid300 & current.ready & current.seconds_from_open.lt(7200) & current.r60.ne(0)
            current=current[eligible].copy()
            current["direction"]=-np.sign(current.r60)
            current["phase"]=current.seconds_from_open.mod(60)
            current["group"]=np.where(current.r60.abs().gt(current.threshold_r60),"extreme","ordinary")
            for horizon in (60,180,300):
                joined=current.merge(future_outcomes(frame,replace(settings,holding_seconds=horizon)),on="timestamp",validate="one_to_one")
                joined["signed_bps"]=joined.direction*joined.delayed_return_bps
                joined["horizon"]=horizon
                panels.append(joined[["date","timestamp","phase","group","horizon","direction","signed_bps"]])
                for (phase,group),rows in joined.groupby(["phase","group"]):
                    values=rows.signed_bps.dropna()
                    daily.append(dict(date=day,phase=phase,group=group,horizon=horizon,signals=len(rows),labelled=len(values),mean_signed_bps=values.mean()))
        history.append(ordinary)
    daily=pd.DataFrame(daily)
    # Average phase means inside each date before bootstrapping dates. Four
    # correlated phases do not create four independent copies of that date.
    pooled=daily.groupby(["date","group","horizon"]).agg(signals=("signals","sum"),labelled=("labelled","sum"),mean_signed_bps=("mean_signed_bps","mean")).reset_index()
    pooled["phase"]="all_phases_equal_weight"
    combined=pd.concat([daily.assign(phase=daily.phase.astype(str)),pooled],ignore_index=True)
    dates=sorted(paths)[5:]
    parts=[("all_explored",dates),("original_discovery",dataset["discovery_dates"]),
           ("original_check",dataset["check_dates"]),("added_reused",dataset["added_dates"]),("backfilled",dataset["backfilled_dates"])]
    rows=[]
    for part,selected_dates in parts:
        for (phase,group,horizon),values in combined[combined.date.isin(selected_dates)].groupby(["phase","group","horizon"]):
            low,high=daily_ci(values.mean_signed_bps)
            rows.append(dict(part=part,phase=phase,group=group,horizon=horizon,days=len(values),signals=int(values.signals.sum()),
                             labelled=int(values.labelled.sum()),positive_days=int(values.mean_signed_bps.gt(0).sum()),
                             mean_signed_bps=values.mean_signed_bps.mean(),ci_low=low,ci_high=high))
    summary=pd.DataFrame(rows)
    daily.to_csv(output/"daily.csv",index=False)
    pooled.to_csv(output/"pooled_by_date.csv",index=False)
    summary.to_csv(output/"summary.csv",index=False)
    pd.concat(panels,ignore_index=True).to_parquet(output/"evaluation.parquet",index=False)
    metadata=dict(study="diagnostic_reversal_all_phases",dataset_manifest_sha256=file_hash(dataset_path),
                  source_manifest=dataset["sources"],code_sha256=file_hash(Path(__file__)),future_labels_used_for_decisions=False,
                  note="All dates explored; phase means pooled within date; bootstrap is descriptive and unadjusted for earlier search.")
    (output/"study.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    shown=summary[summary.part.eq("all_explored")]
    (output/"report.md").write_text("\n".join([
        "# 四个采样相位的正股反转诊断","","只解释正股信号，不生成交易策略，不模拟期权利润。阈值只用此前日期，未来收益只用于评估。",
        "每个日期先平均四个相位，再按日期计算区间。信号有重叠，相位不是独立样本。未校正前面大量探索。",
        "ordinary 是同一时段非极端变动的反向信号，并非精确匹配的因果对照。各相位和日期分组均保留。","",
        "```",shown.to_string(index=False),"```",""]),encoding="utf-8")
    print(shown.to_string(index=False))
    print("Report: "+output.name)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data",type=Path,required=True)
    parser.add_argument("--local-data",type=Path,default=Path("data"))
    parser.add_argument("--dataset",type=Path,default=Path("research/complete_calendar_dataset.json"))
    args=parser.parse_args(argv)
    output=Path("results/research")/("phase_stock_diagnostics_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data,args.local_data,args.dataset,output)


if __name__=="__main__":
    main()
