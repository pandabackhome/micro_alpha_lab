"""Richer stock-only microstructure inputs, past-date option net-PnL targets."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_engine.analysis.causal_volatility import StudySettings, file_hash, read_books
from research_engine.analysis.active_settings import active_settings
from research_engine.analysis.directional_factors import execution_metrics, simulate_single_day
from research_engine.analysis.frozen_dataset import research_inputs
from research_engine.analysis.option_target_search import BUFFERS, RIGHTS, choose_signals, decision_quotes, episode_labels
from research_engine.analysis.stock_prediction_search import FEATURES as BASE_FEATURES, model_features
from research_engine.analysis.stock_strategy_search import COLUMNS, HORIZONS, aggregate, fresh_gate, pick, stock_observations

DIRECT = ["qqq_spread_bps", "depth_imbalance", "depth_imbalance_change", "microprice_delta_bps",
          "depth_imbalance_3s_avg", "depth_imbalance_5s_avg", "depth_imbalance_10s_avg",
          "ret_1s", "ret_3s", "ret_10s", "vol_5s", "vol_10s", "vol_60s", "vol_300s",
          "high_low_range_10s", "high_low_range_30s", "trade_imbalance_change", "trade_imbalance_acceleration"]
DIRECT += ["trade_imbalance_"+str(w)+"s" for w in (1,3,5,10,30)]
OFI = ["ofi_"+str(w)+"s" for w in (1,3,5,10)] + ["ofi_change","ofi_acceleration"]
QUANTITIES = ["bid_size","ask_size"] + ["total_volume_"+str(w)+"s" for w in (1,3,10,30)]
QUANTITIES += [name+"_"+str(w)+"s" for name in ("trade_count","avg_trade_size","max_trade_size") for w in (5,30)]
FEATURES = BASE_FEATURES + DIRECT + ["scaled_"+name for name in OFI] + ["log_"+name for name in QUANTITIES]
MODELS = ("rich_ridge","rich_tree")
DEFAULT_DATASET = Path(__file__).resolve().parents[2]/"research/complete_calendar_dataset.json"


def rich_features(frame, settings):
    result=model_features(stock_observations(frame,settings))
    result=result[result.scheduled].reset_index(drop=True)
    source=frame[["timestamp"]+DIRECT+OFI+QUANTITIES].copy()
    source["timestamp"]=pd.to_datetime(source.timestamp,utc=True).dt.as_unit("ns")
    depth=(source.bid_size+source.ask_size).where(source.bid_size.gt(0)&source.ask_size.gt(0))
    for name in OFI:
        scaled=source[name]/depth
        source["scaled_"+name]=np.sign(scaled)*np.log1p(scaled.abs())
    for name in QUANTITIES:
        source["log_"+name]=np.log1p(source[name].where(source[name].ge(0)))
    extra=DIRECT+["scaled_"+name for name in OFI]+["log_"+name for name in QUANTITIES]
    result=result.merge(source[["timestamp"]+extra],on="timestamp",validate="one_to_one")
    result[FEATURES]=result[FEATURES].replace([np.inf,-np.inf],np.nan)
    return result


def fitted_predictions(current, history, horizon, model_name, min_samples=500):
    if len(history)<5:
        raise ValueError("five previous dates required")
    past=pd.concat(history[-5:],ignore_index=True)
    if past.date.max()>=current.date.min() or past.date.nunique()!=5:
        raise ValueError("five distinct earlier dates required")
    predictions,audits={},[]
    for right in RIGHTS:
        target="target_"+right+"_"+str(horizon)
        eligible=past.valid300 & np.isfinite(past[BASE_FEATURES]).all(axis=1) & np.isfinite(past[target])
        train=past[eligible]
        result=np.full(len(current),np.nan)
        if len(train)>=min_samples:
            if model_name=="rich_ridge":
                model=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler(),Ridge(alpha=1000.))
            elif model_name=="rich_tree":
                model=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),HistGradientBoostingRegressor(
                    max_depth=3,max_iter=150,learning_rate=.05,min_samples_leaf=100,l2_regularization=20.,
                    early_stopping=False,random_state=42))
            else:
                raise ValueError("unknown rich model")
            model.fit(train[FEATURES],train[target])
            good=current.valid300 & np.isfinite(current[BASE_FEATURES]).all(axis=1)
            if good.any():
                result[good.to_numpy()]=model.predict(current.loc[good,FEATURES])
        predictions[right]=result
        audits.append(dict(model=model_name,right=right,horizon=horizon,training_samples=len(train),
                           history_first_date=past.date.min(),history_last_date=past.date.max()))
    return predictions,audits


def reprice_cached_targets(completed,episodes,old_fee,new_fee):
    result=completed.copy()
    delta=new_fee-old_fee
    for right in RIGHTS:
        for horizon in HORIZONS:
            target="target_"+right+"_"+str(horizon)
            subset=episodes[episodes.right.eq(right)&episodes.horizon.eq(horizon)].set_index("timestamp")
            if not subset.index.is_unique:
                raise ValueError("duplicate cached episode target")
            status=result.timestamp.map(subset.status)
            sides=np.select([status.eq("closed"),status.eq("unclosed")],[2,1],0)
            if (result[target].notna()&~status.isin(["closed","unclosed"])).any():
                raise ValueError("finite target without executed episode")
            result[target]=result[target]-sides*delta
    return result


def run(archive_data, local_data, cache_root, output, fee=None):
    paths,dataset=research_inputs(archive_data,local_data,DEFAULT_DATASET)
    cache_meta=json.loads((cache_root/"study.json").read_text())
    cache_sources={entry["file"]:entry["sha256"] for entry in cache_meta["source_manifest"]}
    settings=active_settings(**({"commission_per_contract":fee} if fee is not None else {}))
    output.mkdir(parents=True,exist_ok=False)
    for folder in ("opportunities","new_training_episodes"):
        (output/folder).mkdir()
    history,daily,signals_saved,training,label_counts=[],[],[],[],[]
    for day in sorted(paths):
        origin,feature_path,normalized_path=paths[day]
        frame=pd.read_parquet(feature_path,columns=sorted(set(COLUMNS+DIRECT+OFI+QUANTITIES)))
        current=rich_features(frame,settings)
        if set(current.date)!={day}:
            raise ValueError("feature date mismatch")
        books=read_books(normalized_path,day)
        quotes=decision_quotes(current,books,settings)
        if len(history)>=5:
            ledgers=[]
            for horizon in HORIZONS:
                execution=replace(settings,holding_seconds=horizon)
                for model_name in MODELS:
                    predictions,details=fitted_predictions(current,history,horizon,model_name)
                    training.extend(dict(date=day,**detail) for detail in details)
                    for buffer in BUFFERS:
                        signals=choose_signals(current,quotes,predictions,model_name,buffer)
                        signals["horizon"]=horizon
                        signals["history_last_date"]=history[-1].date.iloc[0]
                        signals_saved.append(signals)
                        ledger=simulate_single_day(signals,books,execution,entry_gate=fresh_gate(books))
                        metrics=execution_metrics(ledger)
                        metrics["entry_rejected"]=int(ledger.status.eq("entry_rejected").sum())
                        daily.append(dict(date=day,factor=model_name+"_net"+str(buffer),horizon=horizon,**metrics))
                        ledgers.append(ledger)
            pd.concat(ledgers,ignore_index=True).to_parquet(output/"opportunities"/(day+".parquet"),index=False)
        # Labels for this date are loaded/generated only after its predictions.
        cache_path=cache_root/"training_features"/(day+".parquet")
        if day in cache_meta["dates"]:
            for name,path in [("features/"+day+".parquet",feature_path),("normalized/"+day+"/events.parquet",normalized_path)]:
                if file_hash(path)!=cache_sources[name]:
                    raise ValueError("cached target input changed")
            cached=pd.read_parquet(cache_path)
            targets=[name for name in cached if name.startswith("target_")]
            completed=current.merge(cached[["timestamp"]+targets],on="timestamp",validate="one_to_one")
            pd.testing.assert_series_equal(completed.timestamp,cached.timestamp,check_names=False)
            episodes=pd.read_parquet(cache_root/"training_episodes"/(day+".parquet"),columns=["timestamp","right","horizon","status"])
            completed=reprice_cached_targets(completed,episodes,cache_meta["settings"]["commission_per_contract"],settings.commission_per_contract)
            counts=pd.read_csv(cache_root/"training_status_counts.csv")
            label_counts.append(counts[counts.date.eq(day)])
        else:
            completed,episodes=episode_labels(current,quotes,books,settings)
            episodes.to_parquet(output/"new_training_episodes"/(day+".parquet"),index=False)
            label_counts.append(episodes.groupby(["date","right","horizon","status"]).size().rename("count").reset_index())
        history.append(completed)
        print("Completed rich stock models "+day,flush=True)
    daily=pd.DataFrame(daily)
    discovery=aggregate(daily,dataset["discovery_dates"])
    selected=pick(discovery)
    parts=[("finalists_check",dataset["check_dates"]),("backfilled_check",dataset["backfilled_dates"]),("added_dates_check",dataset["added_dates"])]
    checks={name:aggregate(daily,dates).merge(selected[["factor","horizon"]],on=["factor","horizon"],how="inner") for name,dates in parts}
    for name,values in [("daily",daily),("discovery",discovery),("selected",selected),("training_audit",pd.DataFrame(training)),
                        ("training_status_counts",pd.concat(label_counts,ignore_index=True))]+list(checks.items()):
        values.to_csv(output/(name+".csv"),index=False)
    pd.concat(signals_saved,ignore_index=True).to_parquet(output/"signals.parquet",index=False)
    package=Path(__file__).resolve().parents[2]
    protocol=package/"research/rich_stock_model_protocol.md"
    (output/"protocol.md").write_bytes(protocol.read_bytes())
    metadata=dict(study="rich_stock_inputs_option_net_targets",models=MODELS,features=FEATURES,trials=12,
                  settings=asdict(settings),dates=sorted(paths),source_manifest=dataset["sources"],
                  discovery_dates=dataset["discovery_dates"],check_dates=dataset["check_dates"],
                  added_dates=dataset["added_dates"],backfilled_dates=dataset["backfilled_dates"],
                  protocol_sha256=file_hash(protocol),dataset_manifest_sha256=file_hash(DEFAULT_DATASET),
                  target_cache=cache_root.name,target_cache_metadata_sha256=file_hash(cache_root/"study.json"),
                  cached_label_fee=cache_meta["settings"]["commission_per_contract"],effective_label_fee=settings.commission_per_contract,
                  code_manifest=[dict(file=str(p.relative_to(package)),sha256=file_hash(p)) for p in sorted((package/"research_engine").rglob("*.py"))])
    (output/"study.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    lines=["# 更完整正股特征预测期权净收益","","每张单边手续费：{}美元；训练目标已按该费用重估。".format(settings.commission_per_contract),
           "固定模型、过去5日训练、12组合。所有日期均按真实顺序，补齐和重复查看日期不称作新留出。",
           "目标保留不可退出的压力损失，不能入场目标缺失，不当作零收益。","","## 前段全部结果","","```",discovery.to_string(index=False),"```",""]
    for name,values in checks.items():
        lines += ["## "+name,"","```",values.to_string(index=False),"```",""]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(discovery.to_string(index=False))
    for name,values in checks.items():
        print(name);print(values.to_string(index=False))
    print("Report: "+output.name)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-data",type=Path,required=True)
    parser.add_argument("--local-data",type=Path,default=Path("data"))
    parser.add_argument("--target-cache",type=Path,default=Path("results/research/option_target_20260918_072322_984529"))
    parser.add_argument("--fee-per-side",type=float,help="override current configured fee; archived run used 0.65")
    args=parser.parse_args(argv)
    output=Path("results/research")/("rich_stock_model_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.archive_data,args.local_data,args.target_cache,output,args.fee_per_side)


if __name__=="__main__":
    main()
