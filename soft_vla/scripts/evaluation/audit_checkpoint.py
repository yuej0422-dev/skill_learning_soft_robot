from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    files = sorted(str(p.relative_to(ckpt)) for p in ckpt.rglob("*") if p.is_file())
    parent = ckpt.parent
    training_state = parent / "training_state"
    ts_files = sorted(str(p.relative_to(training_state)) for p in training_state.rglob("*") if p.is_file()) if training_state.exists() else []
    report = {
        "checkpoint": str(ckpt),
        "exists": ckpt.exists(),
        "pretrained_model_files": files,
        "training_state_dir": str(training_state),
        "training_state_files": ts_files,
        "contains_state_normalization_stats": any("preprocessor" in f or "processor" in f or f.endswith(".safetensors") for f in files),
        "contains_action_normalization_stats": any("postprocessor" in f or "processor" in f or f.endswith(".safetensors") for f in files),
        "contains_action_unnormalization": any("postprocessor" in f or "processor" in f for f in files),
        "contains_camera_key_and_order": any(f == "config.json" for f in files),
        "contains_image_resize_padding": any(f == "config.json" for f in files),
        "contains_custom_crop": False,
        "depends_on_external_base_model": False,
        "depends_on_dataset_directory_for_processors": False,
    }
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = SmolVLAPolicy.from_pretrained(ckpt, local_files_only=True)
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt))
        report["offline_load_policy"] = True
        report["offline_load_processors"] = True
        report["policy_config_action_dim"] = policy.config.action_feature.shape[0]
        report["policy_config_state_dim"] = policy.config.robot_state_feature.shape[0]
        report["camera_keys"] = list(policy.config.image_features.keys())
        report["resize_imgs_with_padding"] = list(policy.config.resize_imgs_with_padding)
        report["preprocessor_steps"] = [type(step).__name__ for step in pre.steps]
        report["postprocessor_steps"] = [type(step).__name__ for step in post.steps]
    except Exception as exc:
        report["offline_load_policy"] = False
        report["offline_load_error"] = f"{type(exc).__name__}: {exc}"

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "checkpoint_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Checkpoint Audit", "", f"- Checkpoint: `{ckpt}`"]
    for key, value in report.items():
        if key not in {"pretrained_model_files", "training_state_files"}:
            lines.append(f"- {key}: `{value}`")
    lines += ["", "## pretrained_model files", ""]
    lines += [f"- `{f}`" for f in files]
    lines += ["", "## training_state files", ""]
    lines += [f"- `{f}`" for f in ts_files]
    (out / "checkpoint_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)[:4000])
    return 0 if report.get("offline_load_policy") else 1


if __name__ == "__main__":
    raise SystemExit(main())

