"""Causal recorded-trade VWAP and longer underlying context for option entries."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.frozen_dataset import research_inputs, DEFAULT_MANIFEST
from research_engine.analysis.stock_strategy_search import COLUMNS, aggregate, fresh_gate, pick, stock_observations

RULES = ("vwap_reclaim", "trend_pullback", "vwap_stretch_fade", "channel_breakout", "opening_retest")


def recorded_vwap(timestamps, trades, opened):
    current = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).as_unit("ns")
    trades = trades.copy()
    trades["available_at"] = pd.to_datetime(trades.available_at, utc=True).dt.as_unit("ns")
    good = trades.available_at.ge(opened) & trades.price.gt(0) & trades.volume.gt(0) & np.isfinite(trades.price) & np.isfinite(trades.volume)
    trades = trades[good].sort_values("available_at", kind="stable")
    values = np.full(len(current), np.nan)
    if len(trades):
        times = pd.DatetimeIndex(trades.available_at).asi8
        vwap = (trades.price * trades.volume).cumsum().to_numpy() / trades.volume.cumsum().to_numpy()
        indices = np.searchsorted(times, current.asi8, side="right") - 1
        known = indices >= 0
        values[known] = vwap[indices[known]]
    return values


def long_observations(frame, trades, settings):
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns")
    result = stock_observations(frame, settings, sampling_seconds=1)
    day = result.date.iloc[0]
    opened = pd.Timestamp(day + " 09:30", tz="America/New_York").tz_convert("UTC")
    valid = (frame.qqq_depth_age_seconds.between(0,5) & frame.qqq_bid.gt(0) & frame.qqq_ask.ge(frame.qqq_bid) &
             np.isfinite(frame.qqq_bid) & np.isfinite(frame.qqq_ask))
    mid = ((frame.qqq_bid + frame.qqq_ask)/2).where(valid)
    vwap = pd.Series(recorded_vwap(frame.timestamp, trades, opened), index=frame.index)
    extra = pd.DataFrame(dict(timestamp=frame.timestamp, vwap=vwap, vwap_deviation=mid/vwap-1,
                               previous_vwap_deviation=(mid/vwap-1).shift(1), r900=mid/mid.shift(900)-1,
                               valid900=valid.rolling(901,min_periods=901).sum().ge(.95*901),
                               high300=mid.shift(1).rolling(300,min_periods=300).max(),
                               low300=mid.shift(1).rolling(300,min_periods=300).min(), previous_spot=mid.shift(1)))
    result = result.merge(extra,on="timestamp",validate="one_to_one")
    result["long_valid"] = result.valid300 & result.valid900 & np.isfinite(result[["r900","vwap_deviation"]]).all(axis=1)
    result["first_opening_up"] = result.seconds_from_open.where(result.spot.gt(result.opening_high)).cummin().ffill()
    result["first_opening_down"] = result.seconds_from_open.where(result.spot.lt(result.opening_low)).cummin().ffill()
    return result


def thresholds(current, history):
    result = current.copy()
    result["long_ready"] = False
    result["history_last_date"] = None
    for field in ("r900", "vwap_deviation", "volume5"):
        result["threshold_"+field] = np.nan
    if len(history) < 5:
        return result
    past = pd.concat(history[-5:], ignore_index=True)
    if past.date.max() >= current.date.min() or past.date.nunique() != 5:
        raise ValueError("five distinct earlier dates required")
    result["history_last_date"] = past.date.max()
    for bucket, idx in result.groupby("half_hour").groups.items():
        group = past[past.half_hour.eq(bucket) & past.long_valid]
        for field in ("r900", "vwap_deviation", "volume5"):
            values = group[field].replace([np.inf,-np.inf],np.nan).dropna()
            if field != "volume5":
                values = values.abs()
            if len(values) >= 100:
                result.loc[idx,"threshold_"+field] = values.quantile(.75)
    result["long_ready"] = result[["threshold_r900","threshold_vwap_deviation","threshold_volume5"]].notna().all(axis=1)
    return result


def long_signals(current):
    p, trend, deviation = current.spot, current.r900, current.vwap_deviation
    strong = trend.abs().gt(current.threshold_r900)
    volume = current.volume5.gt(current.threshold_volume5)
    directions = {
        "vwap_reclaim": np.select([deviation.gt(0) & current.previous_vwap_deviation.le(0) & trend.gt(0),
                                    deviation.lt(0) & current.previous_vwap_deviation.ge(0) & trend.lt(0)], [1,-1], 0),
        "trend_pullback": np.sign(trend).where(strong & (trend*current.r60).lt(0) & (trend*current.r5).gt(0) & (trend*deviation).gt(0), 0),
        "vwap_stretch_fade": -np.sign(deviation).where(deviation.abs().gt(current.threshold_vwap_deviation) & (deviation*current.r5).lt(0), 0),
        "channel_breakout": np.select([p.gt(current.high300) & trend.gt(0), p.lt(current.low300) & trend.lt(0)], [1,-1], 0)* (strong & volume),
        "opening_retest": np.select([
            p.gt(current.opening_high) & current.previous_spot.le(current.opening_high) & trend.gt(0) & (current.seconds_from_open-current.first_opening_up).ge(60),
            p.lt(current.opening_low) & current.previous_spot.ge(current.opening_low) & trend.lt(0) & (current.seconds_from_open-current.first_opening_down).ge(60)], [1,-1], 0),
    }
    rows=[]
    valid = current.long_valid & current.long_ready
    for factor,direction in directions.items():
        result = current.copy()
        sign = pd.Series(direction,index=current.index).fillna(0).astype(int)
        eligible = valid & valid.shift(1).fillna(False) & sign.ne(0)
        triggered = eligible & sign.ne(sign.shift(1).fillna(0))
        result["factor"], result["regime"] = factor, "long_context"
        result["scheduled"] = triggered
        result["direction"] = sign.where(triggered,0)
        result["signal_status"] = np.where(triggered,"signal","no_signal")
        rows.append(result)
    return pd.concat(rows,ignore_index=True)


def run(archive_data, local_data, output):
    settings = StudySettings()
    paths,dataset = research_inputs(archive_data,local_data)
    dates=sorted(paths)
    if len(dates)<16:
        raise ValueError("sixteen dates required")
    output.mkdir(parents=True,exist_ok=False)
    discovery_dates,check_dates,added_dates=dates[5:10],dates[10:14],dates[14:]
    history,daily,ledgers,manifests,observed=[],[],[],[],[]
    for day in dates:
        origin,path,source=paths[day]
        frame=pd.read_parquet(path,columns=COLUMNS)
        trades=pd.read_parquet(source,columns=["available_at","price","volume"],filters=[("kind","=","trade"),("symbol","=","QQQ.US")])
        current=long_observations(frame,trades,settings)
        calibrated=thresholds(current,history)
        history.append(current[current.seconds_from_open.mod(30).eq(0)])
        manifests += [dict(origin=origin,file="features/"+path.name,sha256=file_hash(path)),
                      dict(origin=origin,file="normalized/"+day+"/events.parquet",sha256=file_hash(source))]
        if day not in dates[5:]:
            continue
        signals=long_signals(calibrated)
        books=read_books(source,day)
        observed.append(signals[signals.scheduled])
        for factor in RULES:
            chosen=signals[signals.factor.eq(factor)]
            for horizon in (180,300):
                if not chosen.scheduled.any():
                    # Preserve a no-signal row so daily summaries include inactive dates.
                    chosen=chosen.iloc[:1].copy()
                    chosen["scheduled"]=True
                ledger=simulate_single_day(chosen,books,replace(settings,holding_seconds=horizon),entry_gate=fresh_gate(books))
                ledger["horizon"]=horizon
                metrics=execution_metrics(ledger)
                metrics["entry_rejected"]=int(ledger.status.eq("entry_rejected").sum())
                daily.append(dict(date=day,factor=factor,horizon=horizon,**metrics))
                ledgers.append(ledger)
        print("Evaluated long context "+day,flush=True)
    daily=pd.DataFrame(daily)
    discovery=aggregate(daily,discovery_dates)
    selected=pick(discovery)
    checking=aggregate(daily,check_dates).merge(selected[["factor","horizon"]],on=["factor","horizon"],how="inner")
    added=aggregate(daily,added_dates).merge(selected[["factor","horizon"]],on=["factor","horizon"],how="inner")
    for name,values in [("daily",daily),("discovery",discovery),("selected",selected),("finalists_check",checking),("added_dates_check",added),("all_dates",aggregate(daily,dates[5:]))]:
        values.to_csv(output/(name+".csv"),index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(output/"opportunities.parquet",index=False)
    pd.concat(observed,ignore_index=True).to_parquet(output/"signals.parquet",index=False)
    package=Path(__file__).resolve().parents[2]
    protocol=package/"research/long_context_protocol.md"
    (output/"protocol.md").write_bytes(protocol.read_bytes())
    metadata=dict(study="long_context_stock_rules",rules=RULES,settings=asdict(settings),trials=10,dates=dates,
                  dataset_manifest_sha256=file_hash(DEFAULT_MANIFEST),
                  discovery_dates=discovery_dates,check_dates=check_dates,added_dates=added_dates,
                  source_manifest=manifests,protocol_sha256=file_hash(protocol),
                  code_manifest=[dict(file=str(p.relative_to(package)),sha256=file_hash(p)) for p in sorted((package/"research_engine").rglob("*.py"))])
    (output/"study.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    (output/"report.md").write_text("\n".join([
        "# 成交VWAP、十五分钟趋势与开盘区间回测","","逐秒规则事件，先前日期校准，期权净收益包含全部设置费用。",
        "VWAP仅覆盖本录制的已可用成交。新增两日此前已经查看，不是独立留出。","",
        "## 前段全部结果","","```",discovery.to_string(index=False),"```","",
        "## 候选旧后段","","```",checking.to_string(index=False),"```","",
        "## 候选新增日期","","```",added.to_string(index=False),"```",""]),encoding="utf-8")
    print(discovery.to_string(index=False))
    print("Old check:");print(checking.to_string(index=False))
    print("Added check:");print(added.to_string(index=False))
    print("Report: "+output.name)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data",type=Path,required=True)
    parser.add_argument("--local-data",type=Path,default=Path("data"))
    args=parser.parse_args(argv)
    output=Path("results/research")/("long_context_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data,args.local_data,output)


if __name__=="__main__":
    main()
