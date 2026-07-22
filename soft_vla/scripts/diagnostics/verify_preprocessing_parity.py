from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--dataset-root", default="data/synthetic_soft_robot_vla")
    parser.add_argument("--repo-id", default="local/synthetic_soft_robot_vla")
    args = parser.parse_args()

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    reference = Path(args.reference)
    if not reference.is_absolute():
        reference = PROJECT_ROOT / reference
    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy.config, meta)
    ds = LeRobotDataset(repo_id=args.repo_id, root=root, delta_timestamps=delta_timestamps)
    sample = ds[0]
    processed = pre(sample)
    ref = torch.load(reference, map_location="cpu")
    state_err = (processed["observation.state"].detach().cpu() - ref["state"]).abs().max().item()
    action_err = (processed["action"].detach().cpu() - ref["action"]).abs().max().item()
    image_errs = {}
    for key, ref_img in ref["images"].items():
        image_errs[key] = (processed[key].detach().cpu() - ref_img).abs().max().item()
    max_image_err = max(image_errs.values()) if image_errs else 0.0
    report = {
        "checkpoint": str(checkpoint),
        "reference": str(reference),
        "state_max_abs_error": state_err,
        "action_max_abs_error": action_err,
        "image_max_abs_error": max_image_err,
        "image_errors": image_errs,
        "pass": state_err < 1e-6 and action_err < 1e-6 and max_image_err < 1e-6,
    }
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "train_inference_preprocessing_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "train_inference_preprocessing_parity.md").write_text(
        "# Train/Inference Preprocessing Parity\n\n"
        f"- Checkpoint: `{checkpoint}`\n"
        f"- State max abs error: `{state_err}`\n"
        f"- Action max abs error: `{action_err}`\n"
        f"- Image max abs error: `{max_image_err}`\n"
        f"- PASS: `{report['pass']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
