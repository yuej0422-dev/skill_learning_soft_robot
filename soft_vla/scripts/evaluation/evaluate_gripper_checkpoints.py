from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.config import load_yaml
from soft_vla.data.replay_source import LeRobotReplaySource
from soft_vla.training.gripper import compute_gripper_metrics


DEFAULT_CHECKPOINTS = {
    "baseline": "outputs/gripper_training_comparison/baseline/checkpoints/last/pretrained_model",
    "identity": "outputs/gripper_training_comparison/identity/checkpoints/last/pretrained_model",
    "recommended": "outputs/gripper_training_comparison/recommended/checkpoints/last/pretrained_model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="configs/dataset.real_records.yaml")
    parser.add_argument("--episode-index", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_policy(checkpoint: Path, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def eval_checkpoint(name: str, checkpoint: Path, samples: list[dict], device: str, max_frames: int) -> dict:
    if not checkpoint.exists():
        return {"name": name, "checkpoint": str(checkpoint), "status": "MISSING"}
    policy, preprocessor, postprocessor = load_policy(checkpoint, device)
    preds = []
    gts = []
    for i, sample in enumerate(samples[:max_frames]):
        obs = {k: v for k, v in sample.items() if k not in {"action", "action_is_pad"}}
        batch = preprocessor(obs)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            chunk = policy.predict_action_chunk(batch)
        raw_chunk = postprocessor.process_action(chunk).detach().cpu().numpy()
        preds.append(raw_chunk[0, 0].astype(np.float32))
        gts.append(np.asarray(sample["action"], dtype=np.float32))
    pred = np.stack(preds)
    gt = np.stack(gts)
    err = pred - gt
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "status": "PASS",
        "frames": int(len(pred)),
        "tcp_overall_mae": float(np.mean(np.abs(err[:, :6]))),
        "tcp_per_dim_mae": np.mean(np.abs(err[:, :6]), axis=0).tolist(),
        "gripper": compute_gripper_metrics(pred[:, 6], gt[:, 6]),
        "pred_open_closed_raw_minmax": [float(pred[:, 6].min()), float(pred[:, 6].max())],
    }


def main() -> int:
    args = parse_args()
    ds_cfg = load_yaml(PROJECT_ROOT / args.dataset_config)["dataset"]
    root = Path(ds_cfg["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    samples = list(LeRobotReplaySource(root, ds_cfg["repo_id"], args.episode_index))
    out_root = PROJECT_ROOT / "outputs" / "gripper_training_comparison"
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, rel in DEFAULT_CHECKPOINTS.items():
        ckpt = PROJECT_ROOT / rel
        results[name] = eval_checkpoint(name, ckpt, samples, args.device, args.max_frames)
    (out_root / "evaluation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    lines = ["# Gripper Training Comparison", ""]
    for name, result in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Status: `{result['status']}`")
        lines.append(f"- Checkpoint: `{result['checkpoint']}`")
        if result["status"] == "PASS":
            lines.append(f"- Frames: `{result['frames']}`")
            lines.append(f"- TCP overall MAE: `{result['tcp_overall_mae']}`")
            lines.append(f"- TCP per-dim MAE: `{result['tcp_per_dim_mae']}`")
            lines.append(f"- Gripper metrics: `{result['gripper']}`")
            lines.append(f"- Raw gripper min/max: `{result['pred_open_closed_raw_minmax']}`")
        lines.append("")
    baseline = results.get("baseline", {})
    improved = results.get("recommended", {})
    if baseline.get("status") == "PASS" and improved.get("status") == "PASS":
        lines.extend(
            [
                "## Summary",
                "",
                f"- Gripper baseline F1: `{baseline['gripper']['f1']}`",
                f"- Gripper improved F1: `{improved['gripper']['f1']}`",
                f"- TCP MAE before: `{baseline['tcp_overall_mae']}`",
                f"- TCP MAE after: `{improved['tcp_overall_mae']}`",
                "",
            ]
        )
    (reports / "gripper_training_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
