from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> Path:
    """Make the source tree importable from every real-robot CLI directory."""
    project_root = Path(__file__).resolve().parents[3]
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root
