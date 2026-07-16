from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "soft_vla/outputs/full_runs/smolvla_full_full20000_bs8_20260704_180614/checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET_ROOT = REPO_ROOT / "lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"
DEFAULT_MODES = ["receding_horizon", "temporal_ensemble", "chunk", "single_step"]


def run_cmd(cmd: list[str], *, timeout_s: float | None) -> dict:
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    repo = str(REPO_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([src, repo, existing]) if existing else ":".join([src, repo])
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "ok": False,
            "report": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
            "timeout_s": timeout_s,
        }
    parsed = None
    if proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0 and (parsed or {}).get("ok", proc.returncode == 0),
        "report": parsed,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep SmolVLA four-process deployment modes without touching hardware.")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock chunks instead of loading SmolVLA.")
    parser.add_argument("--real-policy", action="store_true", help="Load the real SmolVLA checkpoint from --checkpoint.")
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="local/soft_robot_7_03_1_delta_tcp")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-inference-chunks", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=None, help="Per-mode subprocess timeout.")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--replan-interval", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.mock == args.real_policy:
        raise SystemExit("Choose exactly one of --mock or --real-policy.")

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else REPO_ROOT / args.dataset_root

    results = []
    for mode in args.modes:
        cmd = [
            sys.executable,
            "soft_vla/scripts/deploy_smolvla_real.py",
            "--mode",
            mode,
            "--duration-s",
            str(args.duration_s),
            "--chunk-size",
            str(args.chunk_size),
            "--execution-horizon",
            str(args.execution_horizon),
            "--replan-interval",
            str(args.replan_interval),
        ]
        if args.mock:
            cmd.append("--mock")
        else:
            cmd.extend(
                [
                    "--real-policy",
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset-root",
                    str(dataset_root),
                    "--repo-id",
                    args.repo_id,
                    "--episode-index",
                    str(args.episode_index),
                    "--video-backend",
                    args.video_backend,
                    "--device",
                    args.device,
                    "--max-inference-chunks",
                    str(args.max_inference_chunks),
                ]
            )
            if args.no_amp:
                cmd.append("--no-amp")
        results.append({"mode": mode, **run_cmd(cmd, timeout_s=args.timeout_s)})

    summary = {
        "ok": all(item["ok"] for item in results),
        "real_policy": args.real_policy,
        "mock": args.mock,
        "modes": args.modes,
        "failed": [item["mode"] for item in results if not item["ok"]],
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
