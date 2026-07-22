from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.schemas import ACTION_NAMES, STATE_NAMES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/deployment_bundle_smolvla")
    parser.add_argument("--merge-lora", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model_out = out / "model"
    shutil.copytree(checkpoint, model_out)

    import lerobot
    import peft
    import torch
    import transformers
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(model_out, local_files_only=True)
    manifest = {
        "policy_type": "smolvla",
        "lerobot_version": getattr(lerobot, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "camera_keys": list(policy.config.image_features.keys()),
        "camera_order": [k.split(".")[-1] for k in policy.config.image_features.keys()],
        "state_dim": 13,
        "action_dim": 7,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "normalization_mapping": {k: str(v) for k, v in policy.config.normalization_mapping.items()},
        "image_crop": {},
        "image_resize": {"resize_imgs_with_padding": list(policy.config.resize_imgs_with_padding)},
        "image_padding": {"method": "LeRobot resize_with_pad"},
        "color_space": "RGB",
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "base_model": "lerobot/smolvla_base",
        "lora_merged": False,
        "dry_run_required": True,
    }
    (out / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "feature_schema.json").write_text(json.dumps({"state_names": STATE_NAMES, "action_names": ACTION_NAMES}, indent=2), encoding="utf-8")
    (out / "image_pipeline.json").write_text(json.dumps(manifest["image_resize"] | {"color_space": "RGB", "custom_crop": None}, indent=2), encoding="utf-8")
    (out / "versions.json").write_text(
        json.dumps({k: manifest[k] for k in ["lerobot_version", "torch_version", "transformers_version", "peft_version"]}, indent=2),
        encoding="utf-8",
    )
    sums = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

