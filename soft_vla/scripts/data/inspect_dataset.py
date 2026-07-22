from __future__ import annotations

import argparse
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.config import load_yaml
from soft_vla.data.dataset_inspector import inspect_dataset, write_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_yaml(cfg_path)
    dataset_cfg = cfg["dataset"]
    root = Path(dataset_cfg["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    result = inspect_dataset(root, dataset_cfg.get("repo_id"), dataset_cfg.get("episodes"))
    write_reports(result, PROJECT_ROOT / "reports")
    print(f"inspection_ok: {result.ok}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

