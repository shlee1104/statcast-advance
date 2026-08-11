"""Configuration loading.

Every tunable number in this project lives in config.yaml, not in code.
This module is the single place that file gets read.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read config.yaml once and cache it for the process lifetime."""
    target = path or CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Could not find {target}. Run from the project root."
        )
    with target.open() as handle:
        return yaml.safe_load(handle)


def get(dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested config value using dot notation.

    >>> get("data.default_season")
    2025
    >>> get("flags.min_n.predictable_count")
    20
    """
    node: Any = load_config()
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(dotted_key: str) -> Path:
    """Resolve a config value that names a path, relative to the project root.

    Keeps the project working no matter which directory it is invoked from.
    """
    raw = get(dotted_key)
    if raw is None:
        raise KeyError(f"No path configured at '{dotted_key}'")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else ROOT / candidate
