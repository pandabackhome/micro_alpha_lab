from dataclasses import replace

import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.position_capacity_search import simulate_capacity


def example(exit_bid):
    t=pd.Timestamp('2026-09-17T14:00:00Z')
    signals=pd.DataFrame(dict(timestamp=[t,t+pd.Timedelta(seconds=60)], date='2026-09-17',factor='x',
                              scheduled=True,signal_status='signal',direction=1,spot=100.,regime='x',
                              session_close=t+pd.Timedelta(hours=1)))
    events=pd.DataFrame([dict(symbol='CALL100',option_right='CALL',option_strike=100.,
                              available_at=t+pd.Timedelta(seconds=s),best_bid=bid,best_ask=bid+.02,
                              best_bid_size=1,best_ask_size=1) for s,bid in [(0,1.),(1,1.),(61,1.),(181,exit_bid),(241,exit_bid)]])
    return signals,ContractBooks(events),replace(StudySettings(),holding_seconds=180)


def test_future_winnings_cannot_fund_second_trade_before_first_exit():
    before=[]
    for exit_bid in [1.,10.]:
        signals,books,settings=example(exit_bid)
        ledger,stats=simulate_capacity(signals,books,settings,capacity=2,capital=150.)
        assert ledger.status.tolist()==['closed','cash_rejected']
        assert stats['max_positions']==1 and stats['cash_rejected']==1
        assert stats['end_cash']-150.==pytest.approx(ledger.net_pnl.sum())
        before.append(ledger.iloc[0].cash_after_entry)
    assert before[0]==before[1]


def test_capacity_two_preserves_two_positions_without_borrowing():
    signals,books,settings=example(1.1)
    one,stats_one=simulate_capacity(signals,books,settings,capacity=1)
    two,stats_two=simulate_capacity(signals,books,settings,capacity=2)
    assert one.status.tolist()==['closed','busy']
    assert two.status.tolist()==['closed','closed']
    assert stats_two['max_positions']==2 and two.cash_after_entry.ge(0).all()
    assert stats_two['max_open_debit']==pytest.approx(two.entry_debit.sum())
    assert stats_two['end_cash']-1000.==pytest.approx(two.net_pnl.sum())
