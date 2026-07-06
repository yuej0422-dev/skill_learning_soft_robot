from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return project_root

