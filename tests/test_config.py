import re
from copy import deepcopy
from pathlib import Path

import pytest

from research_engine.config import CACHE_SCOPES, config_hash

ROOT = Path(__file__).resolve().parents[1]

BASE = {
    "paths": {"recordings": "recordings", "data": "data", "results": "results/research"},
    "underlying": "QQQ.US",
    "timezone": "America/New_York",
    "snapshot_interval": "1s",
    "dedup_mode": "none",
    "session": {"open": "09:30:00", "close": "16:00:00"},
    "features": {"returns": {"windows": [1, 3]}},
    "options": {"relative_strikes": [-1, 0, 1]},
    "labels": {"direction": {"threshold_bps": 5, "vol_column": "vol_30s", "vol_k": 2.0},
               "forward_returns": {"horizons": [30]}},
    "model": {"name": "logistic"},
    "walk_forward": {"train_days": 10},
    "execution": {"mode": "underlying"},
    "exit": {"take_profit_pct": 0.3},
}


def mutated(**path_value):
    """A copy of BASE with one nested key changed, addressed as 'a.b.c'."""
    out = deepcopy(BASE)
    for dotted, value in path_value.items():
        node = out
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return out


@pytest.mark.parametrize("scope", sorted(CACHE_SCOPES))
@pytest.mark.parametrize("dotted", [
    "labels__direction", "labels__forward_returns", "model", "walk_forward",
    "execution", "exit", "paths",
])
def test_sections_no_stage_reads_do_not_invalidate_its_cache(scope, dotted):
    """Editing a label threshold must not cost a rebuild of eighteen days.

    Before the scopes existed every cache keyed on the whole config, so a
    change to `labels` -- which no feature builder reads -- rebuilt every
    feature file to produce byte-identical output.
    """
    changed = mutated(**{dotted: {"sentinel": 1}})
    assert config_hash(changed, scope) == config_hash(BASE, scope)


@pytest.mark.parametrize("scope,dotted", [
    ("normalized", "underlying"), ("normalized", "timezone"),
    ("normalized", "session"), ("normalized", "dedup_mode"),
    ("features", "features"), ("features", "options"),
    ("features", "snapshot_interval"), ("features", "session"),
])
def test_sections_a_stage_reads_do_invalidate_its_cache(scope, dotted):
    """The other half: a scope that misses a dependency serves stale output."""
    changed = mutated(**{dotted: {"sentinel": 1}})
    assert config_hash(changed, scope) != config_hash(BASE, scope)


def test_unscoped_hash_still_covers_everything():
    """The default stays conservative for callers that have not declared a scope."""
    assert config_hash(mutated(labels__direction={"x": 1})) != config_hash(BASE)


def test_scopes_cover_every_config_key_the_stage_modules_reference():
    """Pins the scope lists against what the modules actually read.

    A stage that starts reading a new section and forgets to declare it here
    would silently serve an artefact built under the old setting. Scanning the
    source is coarse -- it sees `config["x"]` and `config.get("x")` -- but it
    catches exactly that omission, which is the failure that matters.
    """
    # The word boundary matters: without it `option_config["relative_strikes"]`
    # reads as a top-level `config["relative_strikes"]`, and the scan reports
    # six sub-keys of `options` as missing dependencies that are in fact
    # already covered by `options` itself.
    pattern = re.compile(r"""(?<![A-Za-z0-9_])config(?:\[|\.get\()["']([a-z_]+)["']""")
    for scope, directory in (("normalized", "ingest"), ("features", "features")):
        referenced = set()
        for path in (ROOT / "research_engine" / directory).glob("*.py"):
            referenced |= set(pattern.findall(path.read_text()))
        # `paths` is resolved through resolve_path, not read for content.
        referenced -= {"paths"}
        declared = set(CACHE_SCOPES[scope])
        if scope == "features":
            # The features stage also consumes everything normalize did.
            declared |= set(CACHE_SCOPES["normalized"])
        missing = referenced - declared
        assert not missing, (
            "{} reads {} but they are not in CACHE_SCOPES[{!r}]; its cache would "
            "not notice them changing".format(directory, sorted(missing), scope)
        )
