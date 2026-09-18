import pandas as pd
import pytest

from research_engine.analysis.fee_sensitivity import reprice


def test_fee_repricing_preserves_spread_and_charges_unclosed_entry_only():
    ledger=pd.DataFrame(dict(status=['closed','unclosed'],entry_fill=[1.02,1.02],exit_fill=[1.10,float('nan')]))
    zero=reprice(ledger,0.)
    standard=reprice(ledger,.65)
    assert zero['realized_net']==pytest.approx(8.)
    assert zero['stress_net']==pytest.approx(-94.)
    assert standard['realized_net']==pytest.approx(6.7)
    assert standard['stress_net']==pytest.approx(-95.95)
    assert standard['breakeven_fee_per_side']==pytest.approx(-94/3)
