from __future__ import annotations

import argparse
import json

import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.config import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)["dataset"]
    repo_id = cfg["repo_id"]
    report = {"repo_id": repo_id, "revision": cfg.get("revision")}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id=repo_id, revision=cfg.get("revision"), files_metadata=True)
        report.update(
            {
                "sha": info.sha,
                "tags": info.tags,
                "siblings": [
                    {"rfilename": s.rfilename, "size": getattr(s, "size", None)}
                    for s in info.siblings[:50]
                ],
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    out = PROJECT_ROOT / "reports" / "public_smoke_dataset_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:4000])
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())

