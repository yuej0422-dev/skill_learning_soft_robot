from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_by_dotted_key(config: dict[str, Any], key: str, value: Any) -> None:
    cur = config
    parts = key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def with_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            set_by_dotted_key(out, key, value)
    return out
