from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.policies.smolvla_adapter import probe_smolvla_api
from soft_vla.utils.device import torch_device_report


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, text=True, capture_output=True).stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> int:
    packages = {}
    for name in ["numpy", "PIL", "yaml", "torch", "lerobot", "transformers", "peft"]:
        spec = importlib.util.find_spec(name)
        packages[name] = spec.origin if spec else None

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "torch": torch_device_report(),
        "nvidia_smi": shell(["nvidia-smi"]),
        "pip_show": shell([sys.executable, "-m", "pip", "show", "lerobot", "torch", "transformers", "peft"]),
    }
    try:
        report["smolvla_api"] = probe_smolvla_api()
    except Exception as exc:
        report["smolvla_api_error"] = f"{type(exc).__name__}: {exc}"

    out = PROJECT_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Environment",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Python version: `{sys.version.split()[0]}`",
        f"- Torch report: `{report['torch']}`",
        f"- SmolVLA API error: `{report.get('smolvla_api_error', 'none')}`",
        "",
        "## Packages",
        "",
    ]
    md.extend([f"- {k}: `{v}`" for k, v in packages.items()])
    (out / "environment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report["torch"], indent=2))
    if "smolvla_api_error" in report:
        print(report["smolvla_api_error"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

