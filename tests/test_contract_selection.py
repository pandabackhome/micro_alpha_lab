import pandas as pd

from research_engine.analysis.causal_volatility import ContractBooks
from research_engine.analysis.contract_selection import StrikeBooks


def test_strike_offsets_are_directional_known_at_signal_and_do_not_fallback():
    t = pd.Timestamp('2026-09-15T14:00:00Z')
    rows = []
    for right in ['CALL','PUT']:
        for strike in [98.,99.,100.,101.,102.]:
            rows.append(dict(symbol=right+str(strike),option_right=right,option_strike=strike,
                             available_at=t+pd.Timedelta(seconds=10 if strike==98. else 0),
                             best_bid=1.,best_ask=1.02,best_bid_size=1,best_ask_size=1))
    books = ContractBooks(pd.DataFrame(rows))
    assert StrikeBooks(books,1.).select_leg(t,100.1,'CALL',.5) == (99.,'CALL99.0')
    assert StrikeBooks(books,1.).select_leg(t,100.1,'PUT',.5) == (101.,'PUT101.0')
    assert StrikeBooks(books,-1.).select_leg(t,100.1,'CALL',.5) == (101.,'CALL101.0')
    assert StrikeBooks(books,2.).select_leg(t,100.1,'CALL',.5) is None
    assert StrikeBooks(books,2.).select_leg(t+pd.Timedelta(seconds=10),100.1,'CALL',.5) == (98.,'CALL98.0')
