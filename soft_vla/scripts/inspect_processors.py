from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.policies.smolvla_train_utils import build_lerobot_train_config, jsonable, load_training_yaml


def _stats_dict(stats, key):
    return {k: jsonable(v) for k, v in stats[key].items()} if key in stats else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smolvla_smoke_8gb.yaml")
    args = parser.parse_args()

    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    raw_cfg = load_training_yaml(PROJECT_ROOT, args.config)
    cfg = build_lerobot_train_config(PROJECT_ROOT, raw_cfg)
    try:
        cfg.validate()
    except FileExistsError:
        pass
    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, dataset_stats=dataset.meta.stats)

    sample = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)))
    processed = pre(sample)
    state_raw = sample["observation.state"]
    action_raw = sample["action"]
    state_mean = dataset.meta.stats["observation.state"]["mean"]
    state_std = dataset.meta.stats["observation.state"]["std"]
    action_mean = dataset.meta.stats["action"]["mean"]
    action_std = dataset.meta.stats["action"]["std"]
    expected_state = (state_raw.to(state_mean.device) - state_mean) / state_std
    expected_action = (action_raw.to(action_mean.device) - action_mean) / action_std
    state_err = (processed["observation.state"].detach().cpu() - expected_state.cpu()).abs().max()
    action_err = (processed["action"].detach().cpu() - expected_action.cpu()).abs().max()

    report = {
        "normalization_mapping": jsonable(policy.config.normalization_mapping),
        "state_stats": _stats_dict(dataset.meta.stats, "observation.state"),
        "action_stats": _stats_dict(dataset.meta.stats, "action"),
        "visual_identity_keys": list(policy.config.image_features.keys()),
        "preprocessor_steps": [type(step).__name__ for step in pre.steps],
        "postprocessor_steps": [type(step).__name__ for step in post.steps],
        "state_normalization_max_abs_error": float(state_err),
        "action_normalization_max_abs_error": float(action_err),
        "max_abs_error_threshold": 1e-5,
        "gripper_state_mean": jsonable(state_mean[..., 12]),
        "gripper_state_std": jsonable(state_std[..., 12]),
        "gripper_action_mean": jsonable(action_mean[..., 6]),
        "gripper_action_std": jsonable(action_std[..., 6]),
        "resize_imgs_with_padding": list(policy.config.resize_imgs_with_padding),
        "custom_crop": None,
        "image_dtype_raw": str(sample["observation.images.main"].dtype),
        "image_shape_raw": list(sample["observation.images.main"].shape),
        "image_range_raw": [float(sample["observation.images.main"].min()), float(sample["observation.images.main"].max())],
        "image_shape_processed_before_model_resize": list(processed["observation.images.main"].shape),
        "image_range_processed_before_model_resize": [
            float(processed["observation.images.main"].min()),
            float(processed["observation.images.main"].max()),
        ],
        "smolvla_model_image_transform": "resize_with_pad to resize_imgs_with_padding, then image * 2 - 1",
    }
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "processor_report_before_training.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    md = [
        "# Processor Report Before Training",
        "",
        f"- Normalization mapping: `{report['normalization_mapping']}`",
        f"- Preprocessor steps: `{report['preprocessor_steps']}`",
        f"- Postprocessor steps: `{report['postprocessor_steps']}`",
        f"- State max abs error: `{report['state_normalization_max_abs_error']}`",
        f"- Action max abs error: `{report['action_normalization_max_abs_error']}`",
        f"- resize_imgs_with_padding: `{report['resize_imgs_with_padding']}`",
        f"- Custom crop: `{report['custom_crop']}`",
        f"- Raw image shape/dtype/range: `{report['image_shape_raw']}`, `{report['image_dtype_raw']}`, `{report['image_range_raw']}`",
        f"- Processed image before model resize: `{report['image_shape_processed_before_model_resize']}`, range `{report['image_range_processed_before_model_resize']}`",
        "",
        "SmolVLA model code performs resize-with-padding inside `prepare_images`, then maps images from `[0, 1]` to `[-1, 1]` for SigLIP.",
    ]
    (out / "processor_report_before_training.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "image_pipeline_report.md").write_text(
        "# Image Pipeline Report\n\n"
        "- Custom ROI crop: `false`\n"
        "- Color space: `RGB`\n"
        f"- Raw image dtype: `{report['image_dtype_raw']}`\n"
        f"- Raw shape: `{report['image_shape_raw']}`\n"
        f"- Raw value range: `{report['image_range_raw']}`\n"
        "- Dataset transform range: `[0, 1]` tensor, channel-first with observation dimension.\n"
        f"- SmolVLA resize with padding: `{report['resize_imgs_with_padding']}`\n"
        "- Padding: left/top pad as implemented by LeRobot `resize_with_pad`.\n"
        "- VLM normalization: model maps `[0, 1]` to `[-1, 1]`.\n"
        "- Training/inference crop fork: `false`\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(report), indent=2)[:4000])
    return 0 if float(state_err) < 1e-5 and float(action_err) < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())

