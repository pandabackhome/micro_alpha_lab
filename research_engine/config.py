from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "research_config.yaml"


def portable_path(path) -> str:
    """Store project-relative paths or external filenames, never host home paths."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return resolved.name


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path.resolve())
    return config


#: The config keys each cached stage's *content* actually depends on.
#:
#: Hashing the whole config made every cache a cache of everything: editing
#: `labels.direction` -- which no feature builder reads -- invalidated all
#: eighteen days of features and cost a forty-minute rebuild to produce
#: byte-identical files. Scoping the key to what the stage reads makes a label
#: experiment cost what it should, which is the labels.
#:
#: The lists are a claim about the code and have to be kept true: a stage that
#: starts reading a new config section must gain it here, or it will serve a
#: stale artefact after that section changes. `test_config.py` pins them
#: against what the modules actually reference. Labels appear in no list on
#: purpose -- they are rebuilt every run and cached nowhere.
CACHE_SCOPES: Dict[str, tuple] = {
    "normalized": ("underlying", "timezone", "session", "dedup_mode"),
    "features": ("underlying", "timezone", "session", "dedup_mode",
                 "snapshot_interval", "features", "options"),
}


def config_hash(config: Dict[str, Any], scope: Optional[str] = None) -> str:
    """Digest of the configuration, or of the part `scope` is allowed to see.

    With no scope this hashes everything, which is the safe answer when a
    caller has not said what it depends on. Callers that cache an artefact
    should name their scope instead -- see :data:`CACHE_SCOPES`.
    """
    clean = deepcopy(config)
    clean.pop("_config_path", None)
    if scope is not None:
        keys = CACHE_SCOPES[scope]
        # `paths` is deliberately absent from every scope: it decides where an
        # artefact is written, never what is in it, and including it would make
        # moving the data directory look like a content change.
        clean = {k: clean[k] for k in keys if k in clean}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key]).expanduser()
    if value.is_absolute():
        return value
    config_path = Path(config.get("_config_path", DEFAULT_CONFIG)).resolve()
    return config_path.parent / value
