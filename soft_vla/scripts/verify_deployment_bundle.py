from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.hardware.null_controller import NullRobotController
from soft_vla.hardware.safety_filter import SafetyFilter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="outputs/deployment_bundle_smolvla")
    parser.add_argument("--dataset-root", default="data/synthetic_soft_robot_vla")
    parser.add_argument("--repo-id", default="local/synthetic_soft_robot_vla")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = PROJECT_ROOT / bundle
    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model_dir = bundle / "model"
    policy = SmolVLAPolicy.from_pretrained(model_dir, local_files_only=True)
    policy.to("cuda")
    policy.eval()
    policy.reset()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    ds = LeRobotDataset(repo_id=args.repo_id, root=root)
    sample = ds[0]
    obs = {k: v for k, v in sample.items() if k != "action" and k != "action_is_pad"}
    batch = pre(obs)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        chunk = policy.predict_action_chunk(batch)
    raw = post.process_action(chunk).detach().cpu().numpy()
    action = SafetyFilter().filter_action(raw[0, 0].astype(np.float32))
    ctrl = NullRobotController()
    ctrl.send_action(action)
    report = {
        "bundle": str(bundle),
        "offline_env": {"HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"], "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"]},
        "policy_loaded": True,
        "preprocessor_loaded": True,
        "postprocessor_loaded": True,
        "action_chunk_shape": list(raw.shape),
        "action_shape": list(action.shape),
        "finite": bool(np.isfinite(action).all()),
        "null_controller_actions": len(ctrl.recorded_actions),
        "no_gt_action_input": True,
    }
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "deployment_bundle_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "deployment_bundle_verification.md").write_text(
        "# Deployment Bundle Verification\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in report.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["finite"] and report["null_controller_actions"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())

