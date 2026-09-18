from dataclasses import replace

import pandas as pd
import pytest

from research_engine.analysis.adaptive_exit_search import underlying_exit_scheduler
from research_engine.analysis.causal_volatility import ContractBooks, StudySettings
from research_engine.analysis.directional_factors import simulate_single_day
from research_engine.analysis.reversal_execution_audit import audit_raw_fills
from research_engine.analysis.stock_strategy_search import fresh_gate


def test_first_stock_crossing_has_route_delay_and_is_unchanged_by_later_prices():
    t = pd.Timestamp('2026-09-15T14:00:00Z')
    frame = pd.DataFrame(dict(timestamp=pd.date_range(t, periods=400, freq='1s'),
                              qqq_bid=99.99, qqq_ask=100.01, qqq_depth_age_seconds=0.))
    frame.loc[40:45, ['qqq_bid', 'qqq_ask']] += 1.
    signal = dict(spot=100., r60=100./102.-1., direction=1)
    entry, deadline = t+pd.Timedelta(seconds=1), t+pd.Timedelta(seconds=301)
    before, details = underlying_exit_scheduler(frame, .5, .5)(signal, 'CALL100', entry, deadline)
    assert before == t+pd.Timedelta(seconds=41) and details['exit_reason'] == 'take'
    frame.loc[50:, ['qqq_bid','qqq_ask']] -= 50.
    after, after_details = underlying_exit_scheduler(frame, .5, .5)(signal, 'CALL100', entry, deadline)
    assert before == after and details == after_details
    frame.loc[40:45, 'qqq_depth_age_seconds'] = 100.
    stopped, details = underlying_exit_scheduler(frame, .5, .5)(signal, 'CALL100', entry, deadline)
    assert stopped == t+pd.Timedelta(seconds=51) and details['exit_reason'] == 'stop'


def test_earlier_exit_releases_position_and_independent_raw_audit_detects_bad_price():
    t = pd.Timestamp('2026-09-15T14:00:00Z')
    signals = pd.DataFrame(dict(timestamp=[t,t+pd.Timedelta(seconds=60)], date='2026-09-15', factor='x',
                               scheduled=True, signal_status='signal', direction=1, spot=100., regime='x',
                               session_close=t+pd.Timedelta(hours=1)))
    events = pd.DataFrame([dict(symbol='CALL100', option_right='CALL', option_strike=100.,
                               available_at=t+pd.Timedelta(seconds=s), best_bid=1., best_ask=1.02,
                               best_bid_size=1, best_ask_size=1) for s in [0,1,41,60,61,101]])
    books = ContractBooks(events)
    settings = replace(StudySettings(), holding_seconds=300)
    def schedule(signal, symbol, entry, deadline):
        return entry+pd.Timedelta(seconds=40), {'exit_reason':'test'}
    ledger = simulate_single_day(signals, books, settings, entry_gate=fresh_gate(books), exit_scheduler=schedule)
    assert ledger.status.tolist() == ['closed','closed']
    assert ledger.holding_seconds.eq(40).all()
    assert audit_raw_fills(ledger, events, settings) == 4
    bad = ledger.copy()
    bad.loc[0,'exit_bid'] = 10.
    with pytest.raises(ValueError, match='price or age'):
        audit_raw_fills(bad, events, settings)
    with pytest.raises(ValueError, match='original deadline'):
        simulate_single_day(signals, books, settings, exit_scheduler=lambda s,c,e,d: (d+pd.Timedelta(seconds=1), {}))
