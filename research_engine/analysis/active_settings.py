"""Current user execution costs, separate from archived study defaults."""
from dataclasses import replace

from research_engine.analysis.causal_volatility import StudySettings
from research_engine.config import load_config


def active_settings(**overrides):
    execution=load_config().get("execution",{})
    values=dict(commission_per_contract=float(execution.get("commission_per_contract",1.5)),
                slippage_bps=float(execution.get("slippage_bps",1.)),
                multiplier=int(execution.get("option_multiplier",100)))
    values.update(overrides)
    return replace(StudySettings(),**values)
