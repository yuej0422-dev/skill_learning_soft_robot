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
from soft_vla.policies.smolvla_adapter import probe_smolvla_api


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--single-batch", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    try:
        api = probe_smolvla_api()
    except Exception as exc:
        report = PROJECT_ROOT / "reports" / "smolvla_smoke_status.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# SmolVLA Smoke Status\n\n"
            f"Status: blocked by missing/runtime dependency.\n\n"
            f"Error: `{type(exc).__name__}: {exc}`\n\n"
            "Use `environment.cuda.yml` or install `lerobot[smolvla,peft]`, "
            "`transformers`, and `peft` into a CUDA Torch environment.\n",
            encoding="utf-8",
        )
        print(f"SmolVLA smoke blocked: {type(exc).__name__}: {exc}")
        return 2

    report = PROJECT_ROOT / "reports" / "smolvla_smoke_status.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# SmolVLA Smoke Status\n\n"
        "Status: API import succeeded. Full official training should be wired to "
        "the installed LeRobot trainer with the probed signatures below.\n\n"
        f"Config: `{cfg}`\n\n"
        f"API: `{api}`\n",
        encoding="utf-8",
    )
    print("SmolVLA API import succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

