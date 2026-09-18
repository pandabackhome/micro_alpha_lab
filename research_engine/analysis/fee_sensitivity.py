"""Exact fee repricing of frozen stock rules whose decisions do not use fees."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_engine.analysis.causal_volatility import file_hash

FEES = (0., .15, .35, .65, 1.50)


def reprice(group, fee, multiplier=100):
    closed=group[group.status.eq("closed")]
    unclosed=group[group.status.eq("unclosed")]
    prefee=float(((closed.exit_fill-closed.entry_fill)*multiplier).sum()) if len(closed) else 0.
    unresolved=float((unclosed.entry_fill*multiplier).sum()) if len(unclosed) else 0.
    fee_sides=2*len(closed)+len(unclosed)
    stress_zero=prefee-unresolved
    return dict(closed=len(closed),unclosed=len(unclosed),fee_per_side=fee,
                realized_net=prefee-2*len(closed)*fee,stress_net=stress_zero-fee_sides*fee,
                mean_closed_net=(prefee-2*len(closed)*fee)/len(closed) if len(closed) else None,
                breakeven_fee_per_side=stress_zero/fee_sides if fee_sides else None)


def run(calendar_root,new_root,output):
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name,root in [("complete_calendar",calendar_root),("initial_new_dates",new_root)]:
        meta=json.loads((root/"study.json").read_text())
        ledger=pd.read_parquet(root/"opportunities.parquet")
        groups=meta.get("groups",{"initial_new_dates":meta.get("dates",[])})
        for part,dates in groups.items():
            for factor,panel in ledger[ledger.date.isin(dates)].groupby("factor"):
                for fee in FEES:
                    rows.append(dict(source=name,part=part,factor=factor,**reprice(panel,fee,meta["settings"]["multiplier"])))
    values=pd.DataFrame(rows)
    values.to_csv(output/"fees.csv",index=False)
    metadata=dict(study="fee_sensitivity_frozen_stock_rules",fees=FEES,
                  calendar_study=calendar_root.name,calendar_metadata_sha256=file_hash(calendar_root/"study.json"),
                  new_study=new_root.name,new_metadata_sha256=file_hash(new_root/"study.json"),
                  code_sha256=file_hash(Path(__file__)),note="Fixed signals and fills; these stock-only rules do not use fees to choose entries. Bid/ask spreads and slippage retained.")
    (output/"study.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    (output/"report.md").write_text("\n".join([
        "# 原样股票规则的手续费敏感性","","只改变单边手续费，不假设买卖中价成交。价差、滑点和失败状态保留。",
        "这三个正股规则的入场不使用手续费，故可保持原成交逐笔重算。模型或成本门槛策略不能直接套用同样做法。",
        "盈亏平衡手续费为样本事后描述，负值表示即使手续费为零仍不盈利，不代表未来可承受费用。","",
        "```",values.to_string(index=False),"```",""]),encoding="utf-8")
    print(values[values.part.isin(["initial_new_dates","added_reused"])].to_string(index=False))
    print("Report: "+output.name)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-root",type=Path,required=True)
    parser.add_argument("--new-root",type=Path,required=True)
    args=parser.parse_args(argv)
    output=Path("results/research")/("fee_sensitivity_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run(args.calendar_root,args.new_root,output)


if __name__=="__main__":
    main()
