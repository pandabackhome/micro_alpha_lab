from __future__ import annotations

import operator
import re
from typing import Dict, Sequence

import pandas as pd


COMPARATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt,
               "<=": operator.le, "==": operator.eq, "!=": operator.ne}
CONDITION = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")


def evaluate_rules(dataset: pd.DataFrame, rules: Sequence[Dict], label: str = "future_ret_30s") -> pd.DataFrame:
    rows = []
    for rule in rules:
        mask = pd.Series(True, index=dataset.index)
        for name, expression in rule.get("conditions", {}).items():
            if name not in dataset:
                raise KeyError("rule feature missing: {}".format(name))
            parsed = CONDITION.fullmatch(str(expression))
            if parsed is None:
                raise ValueError("unsupported rule condition: {}".format(expression))
            mask &= COMPARATORS[parsed.group(1)](dataset[name], float(parsed.group(2))).fillna(False)
        matches = dataset.loc[mask]
        returns = matches[label].dropna()
        rows.append({
            "rule": rule["name"], "signal_count": len(matches),
            "future_return": returns.mean(), "win_rate": (returns > 0).mean() if len(returns) else float("nan"),
            "mfe": matches["mfe_30s"].mean() if "mfe_30s" in matches else float("nan"),
            "mae": matches["mae_30s"].mean() if "mae_30s" in matches else float("nan"),
        })
    return pd.DataFrame(rows)

