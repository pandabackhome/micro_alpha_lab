from dataclasses import replace

import pandas as pd
import pytest

from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.vertical_spread_search import metrics, simulate_spreads


def spread_case(with_exit=True, future_protection=False):
    t = pd.Timestamp('2026-09-17T14:00:00Z')
    signals = pd.DataFrame(dict(timestamp=[t,t+pd.Timedelta(seconds=60)], date='2026-09-17', factor='test',
                               scheduled=True, signal_status='signal', direction=1, spot=100.,
                               session_close=t+pd.Timedelta(hours=1)))
    events = []
    for symbol,strike,bid in [('P100',100.,.6),('P99',99.,.2)]:
        for second in ([1] if future_protection and symbol=='P99' else [0,1]):
            events.append(dict(symbol=symbol,option_right='PUT',option_strike=strike,
                               available_at=t+pd.Timedelta(seconds=second),best_bid=bid,best_ask=bid+.02,
                               best_bid_size=1,best_ask_size=1))
        if with_exit:
            bid = .30 if symbol=='P100' else .10
            events.append(dict(symbol=symbol,option_right='PUT',option_strike=strike,
                               available_at=t+pd.Timedelta(seconds=181),best_bid=bid,best_ask=bid+.02,
                               best_bid_size=1,best_ask_size=1))
    return signals,ContractBooks(pd.DataFrame(events)),replace(StudySettings(),holding_seconds=180)


def test_credit_spread_cash_flows_include_both_protection_and_four_fees():
    signals,books,settings=spread_case()
    ledger=simulate_spreads(signals,books,settings)
    assert ledger.status.tolist()==['closed','busy']
    row=ledger.iloc[0]
    entry=(.60*.9999-.22*1.0001)*100-1.30
    exit_cost=(.32*1.0001-.10*.9999)*100+1.30
    assert row.entry_credit==pytest.approx(entry)
    assert row.exit_debit==pytest.approx(exit_cost)
    assert row.net_pnl==pytest.approx(entry-exit_cost)
    assert row.maximum_loss_model==pytest.approx(100-entry+1.30)
    assert row.commission==2.60 and row.short_symbol=='P100' and row.long_symbol=='P99'


def test_unclosed_spread_blocks_day_and_cannot_use_future_protection_contract():
    signals,books,settings=spread_case(with_exit=False)
    ledger=simulate_spreads(signals,books,settings)
    assert ledger.status.tolist()==['unclosed','busy']
    result=metrics(ledger)
    assert result['realized_net_pnl']==0.
    assert result['net_with_unclosed_zero_recovery']==pytest.approx(-ledger.iloc[0].maximum_loss_model)
    signals,books,settings=spread_case(future_protection=True)
    assert simulate_spreads(signals,books,settings).iloc[0].status=='missing_contract'
