from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()
REPO_ROOT = PROJECT_ROOT.parent


def run_cmd(cmd: list[str], *, cwd: Path) -> dict:
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    repo = str(REPO_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([src, repo, existing]) if existing else ":".join([src, repo])
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all non-hardware soft robot control validation checks.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-unittest", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    results: list[dict] = []
    if not args.skip_unittest:
        results.append(
            run_cmd(
                [py, "-m", "unittest", "discover", "-s", "tests"],
                cwd=PROJECT_ROOT,
            )
        )
    results.append(
        run_cmd(
            [py, "soft_vla/scripts/inspect_real_control_assets.py", "--episode-index", str(args.episode_index)],
            cwd=REPO_ROOT,
        )
    )
    results.append(
        run_cmd(
            [py, "soft_vla/scripts/test_pressure_driver.py", "--packet-channels", "16", "--pressure", "0.1"],
            cwd=REPO_ROOT,
        )
    )
    results.append(
        run_cmd(
            [
                py,
                "soft_vla/scripts/debug_single_point_target_real.py",
                "--mock",
                "--target-delta",
                "0.001,0,0,0,0,0",
                "--duration-s",
                "0.1",
                "--feedforward",
                "zero",
                "--feedback",
                "none",
            ],
            cwd=REPO_ROOT,
        )
    )
    results.append(
        run_cmd(
            [
                py,
                "soft_vla/scripts/deploy_smolvla_real.py",
                "--mock",
                "--mode",
                "receding_horizon",
                "--duration-s",
                "0.5",
            ],
            cwd=REPO_ROOT,
        )
    )
    results.append(
        run_cmd(
            [
                py,
                "soft_vla/scripts/replay_episode_real.py",
                "--mock",
                "--episode-index",
                str(args.episode_index),
                "--max-frames",
                "2",
                "--feedforward",
                "pressure_model",
                "--feedback",
                "integral_lqr",
                "--device",
                args.device,
                "--delta-tcp-scale",
                "0.1",
                "--pressure-scale",
                "0.2",
                "--feedback-gain-scale",
                "0.05",
            ],
            cwd=REPO_ROOT,
        )
    )
    with tempfile.TemporaryDirectory(prefix="soft_vla_nonhardware_") as tmp:
        fixed_k = Path(tmp) / "fixed_k_integral.npz"
        results.append(
            run_cmd(
                [
                    py,
                    "soft_vla/scripts/build_fixed_k_integral.py",
                    "--output",
                    str(fixed_k),
                    "--device",
                    args.device,
                ],
                cwd=REPO_ROOT,
            )
        )
        combos = [
            ("pressure_model", "integral_lqr", None),
            ("pressure_model", "fixed_k_integral", fixed_k),
            ("awac", "integral_lqr", None),
            ("awac", "fixed_k_integral", fixed_k),
        ]
        for feedforward, feedback, gain_path in combos:
            cmd = [
                py,
                "soft_vla/scripts/dry_run_episode_motion_control.py",
                "--episode-index",
                str(args.episode_index),
                "--max-frames",
                str(args.max_frames),
                "--feedforward",
                feedforward,
                "--feedback",
                feedback,
                "--device",
                args.device,
                "--feedback-gain-scale",
                "0.05",
                "--pressure-scale",
                "0.3",
            ]
            if gain_path is not None:
                cmd.extend(["--fixed-k-path", str(gain_path)])
            results.append(run_cmd(cmd, cwd=REPO_ROOT))

    ok = all(item["ok"] for item in results)
    summary = {
        "ok": ok,
        "checks": len(results),
        "failed": [item["cmd"] for item in results if not item["ok"]],
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
