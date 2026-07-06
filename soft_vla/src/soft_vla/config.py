from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(base_dir) / p

