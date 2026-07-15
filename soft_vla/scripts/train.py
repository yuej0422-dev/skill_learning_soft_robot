from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.policies.smolvla_train_utils import (
    build_lerobot_train_config,
    first_trainable_parameter,
    jsonable,
    load_training_yaml,
    parameter_summary,
)
from soft_vla.training.gripper import (
    apply_identity_stats_for_indices,
    apply_hybrid_action_stats,
    build_transition_weights,
    smolvla_weighted_action_loss,
)


def tensor_checksum(t: torch.Tensor) -> float:
    return float(t.detach().float().sum().cpu())


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def training_mode_label(cfg) -> str:
    if cfg.peft is not None:
        return "lora fine-tuning"
    if not bool(cfg.policy.train_expert_only) and not bool(cfg.policy.freeze_vision_encoder):
        return "full-parameter fine-tuning"
    if bool(cfg.policy.train_expert_only):
        return "expert/action-head fine-tuning"
    return "partial fine-tuning"


def write_gripper_sampler_report(path: Path, report, sampled: dict) -> None:
    lines = [
        "# Gripper Sampler Report",
        "",
        f"- Raw transition-window frame ratio: `{report.raw_transition_frame_ratio}`",
        f"- Weighted transition-window mass ratio: `{report.weighted_transition_mass_ratio}`",
        f"- Dataset open frame ratio: `{report.open_frame_ratio}`",
        f"- Dataset closed frame ratio: `{report.closed_frame_ratio}`",
        f"- Transition indices: `{report.transition_indices[:80]}`",
        f"- Episode transition counts: `{report.episode_transition_counts}`",
        "",
        "## Actual Training Sample Mix",
        "",
        f"- Samples seen: `{sampled.get('samples', 0)}`",
        f"- Sampled transition-window ratio: `{sampled.get('transition_ratio', 0.0)}`",
        f"- Sampled open ratio: `{sampled.get('open_ratio', 0.0)}`",
        f"- Sampled closed ratio: `{sampled.get('closed_ratio', 0.0)}`",
        f"- Episode sampled counts: `{sampled.get('episode_counts', {})}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_gripper_normalization_report(path: Path, dataset_stats: dict, processor_stats: dict, first_report: dict) -> None:
    action_stats = dataset_stats["action"]
    patched_action_stats = processor_stats["action"]
    lines = [
        "# Gripper Normalization Report",
        "",
        "- TCP normalization mode: `MEAN_STD`",
        "- Gripper normalization mode: `IDENTITY` via action mean/std patch",
        f"- TCP mean: `{patched_action_stats['mean'][:6]}`",
        f"- TCP std: `{patched_action_stats['std'][:6]}`",
        f"- Original gripper mean/std: `{action_stats['mean'][6]}`, `{action_stats['std'][6]}`",
        f"- Processor gripper mean/std: `{patched_action_stats['mean'][6]}`, `{patched_action_stats['std'][6]}`",
        f"- First processed action shape: `{first_report.get('processed_action_shape')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_identity_indices(raw_cfg: dict) -> dict[str, list[int]]:
    overrides = raw_cfg.get("normalization_overrides", {})
    identity_indices: dict[str, list[int]] = {}
    for feature_key, feature_cfg in overrides.items():
        indices = feature_cfg.get("identity_indices", [])
        if indices:
            identity_indices[str(feature_key)] = [int(i) for i in indices]
    return identity_indices


def apply_image_transforms(batch: dict, raw_cfg: dict) -> dict:
    cfg = raw_cfg.get("image_transforms", {})
    crop_cfg = cfg.get("crop_right_fraction", {})
    if not crop_cfg:
        return batch
    out = dict(batch)
    for key, fraction in crop_cfg.items():
        if key not in out:
            continue
        value = out[key]
        if not hasattr(value, "shape") or value.ndim < 4:
            continue
        width = int(value.shape[-1])
        keep_width = int(round(width * (1.0 - float(fraction))))
        if keep_width <= 0 or keep_width > width:
            raise ValueError(f"Invalid crop_right_fraction={fraction} for {key} width={width}")
        out[key] = value[..., :keep_width].contiguous()
    return out


def write_mixed_normalization_report(path: Path, dataset_stats: dict, processor_stats: dict, identity_indices: dict[str, list[int]], first_report: dict) -> None:
    lines = [
        "# Mixed Normalization Report",
        "",
        "- Non-identity dimensions keep dataset mean/std normalization.",
        "- Configured identity dimensions use processor mean=0/std=1.",
        f"- Identity indices: `{identity_indices}`",
        f"- First processed state shape: `{first_report.get('processed_state_shape')}`",
        f"- First processed action shape: `{first_report.get('processed_action_shape')}`",
        "",
    ]
    for feature_key, indices in identity_indices.items():
        original = dataset_stats[feature_key]
        patched = processor_stats[feature_key]
        original_mean = [float(original["mean"][i]) for i in indices]
        original_std = [float(original["std"][i]) for i in indices]
        patched_mean = [float(patched["mean"][i]) for i in indices]
        patched_std = [float(patched["std"][i]) for i in indices]
        lines.extend(
            [
                f"## {feature_key}",
                "",
                f"- Original mean at identity indices: `{original_mean}`",
                f"- Original std at identity indices: `{original_std}`",
                f"- Processor mean at identity indices: `{patched_mean}`",
                f"- Processor std at identity indices: `{patched_std}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from lerobot.datasets.factory import make_dataset
    from lerobot.datasets.utils import cycle
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint

    raw_cfg = load_training_yaml(PROJECT_ROOT, args.config)
    cfg = build_lerobot_train_config(PROJECT_ROOT, raw_cfg)
    if cfg.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{cfg.output_dir} exists. Use --overwrite.")
        shutil.rmtree(cfg.output_dir)
    cfg.validate()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    command = f"{Path(__file__).name} --config {args.config} --overwrite"
    write_shared_reports = bool(raw_cfg.get("reports", {}).get("update_shared", True))
    (cfg.output_dir / "actual_training_command.txt").write_text(command + "\n", encoding="utf-8")
    if write_shared_reports:
        (PROJECT_ROOT / "reports" / "actual_training_command.txt").write_text(command + "\n", encoding="utf-8")

    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if cfg.peft is not None:
        policy = policy.wrap_with_peft(peft_cli_overrides=cfg.peft.__dict__)
    identity_indices = parse_identity_indices(raw_cfg)
    action_norm_cfg = raw_cfg.get("action_normalization", {})
    use_hybrid_action_norm = (
        str(action_norm_cfg.get("gripper_mode", "")).lower() == "identity"
        or str(action_norm_cfg.get("gripper", {}).get("mode", "")).lower() == "identity"
    )
    if identity_indices:
        processor_stats = apply_identity_stats_for_indices(dataset.meta.stats, identity_indices)
    elif use_hybrid_action_norm:
        processor_stats = apply_hybrid_action_stats(dataset.meta.stats)
    else:
        processor_stats = dataset.meta.stats

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "normalizer_processor": {
                "stats": processor_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": processor_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    gripper_cfg = raw_cfg.get("gripper_training", {})
    sampler = None
    sampler_report = None
    transition_mask_for_index = None
    oversample_cfg = gripper_cfg.get("transition_oversampling", {})
    if bool(oversample_cfg.get("enabled", False)):
        weights, sampler_report = build_transition_weights(
            dataset,
            enabled=True,
            before_steps=int(oversample_cfg.get("before_steps", 5)),
            after_steps=int(oversample_cfg.get("after_steps", 5)),
            transition_weight=float(oversample_cfg.get("transition_weight", 4.0)),
            normal_weight=float(oversample_cfg.get("normal_weight", 1.0)),
        )
        from soft_vla.training.gripper import extract_dataset_arrays, transition_window_mask

        actions_np, episodes_np, _frames_np, _indices_np = extract_dataset_arrays(dataset)
        transition_mask_for_index = transition_window_mask(
            actions_np,
            episodes_np,
            before_steps=int(oversample_cfg.get("before_steps", 5)),
            after_steps=int(oversample_cfg.get("after_steps", 5)),
        )[0]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=max(len(dataset), cfg.steps * cfg.batch_size),
            replacement=True,
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=cfg.policy.device == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(loader)

    raw_first = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)))
    raw_first = apply_image_transforms(raw_first, raw_cfg)
    processed_first = preprocessor(raw_first)
    raw_processed_report = {
        "raw_batch_keys": list(raw_first.keys()),
        "raw_image_shapes": {k: list(v.shape) for k, v in raw_first.items() if k.startswith("observation.images") and hasattr(v, "shape")},
        "raw_state_shape": list(raw_first["observation.state"].shape),
        "raw_action_shape": list(raw_first["action"].shape),
        "task_example": raw_first.get("task"),
        "processed_batch_keys": list(processed_first.keys()),
        "processed_image_shapes": {
            k: list(v.shape) for k, v in processed_first.items() if k.startswith("observation.images") and hasattr(v, "shape")
        },
        "processed_state_shape": list(processed_first["observation.state"].shape),
        "processed_action_shape": list(processed_first["action"].shape),
        "processed_dtype": str(processed_first["observation.state"].dtype),
        "processed_device": str(processed_first["observation.state"].device),
    }
    (cfg.output_dir / "first_batch_report.json").write_text(json.dumps(jsonable(raw_processed_report), indent=2), encoding="utf-8")
    if identity_indices:
        write_mixed_normalization_report(
            PROJECT_ROOT / "reports" / "mixed_normalization_report.md",
            dataset.meta.stats,
            processor_stats,
            identity_indices,
            raw_processed_report,
        )
    elif use_hybrid_action_norm:
        write_gripper_normalization_report(
            PROJECT_ROOT / "reports" / "gripper_normalization_report.md",
            dataset.meta.stats,
            processor_stats,
            raw_processed_report,
        )
    torch.save(
        {
            "state": processed_first["observation.state"].detach().cpu(),
            "action": processed_first["action"].detach().cpu(),
            "images": {k: v.detach().cpu() for k, v in processed_first.items() if k.startswith("observation.images") and not k.endswith("_is_pad")},
        },
        cfg.output_dir / "preprocessing_parity_reference.pt",
    )

    summary = parameter_summary(policy)
    param_name, param = first_trainable_parameter(policy)
    before = param.detach().float().cpu().clone()
    metrics = []
    torch.cuda.reset_peak_memory_stats()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    start_all = time.perf_counter()
    loss_weight_cfg = gripper_cfg.get("loss_weight", {})
    use_weighted_loss = bool(loss_weight_cfg.get("enabled", False))
    tcp_loss_weight = float(loss_weight_cfg.get("tcp_weight", 1.0))
    gripper_loss_weight = float(loss_weight_cfg.get("gripper_weight", 1.0))
    sampled_indices: list[int] = []
    sampled_gripper: list[float] = []
    sampled_episodes: list[int] = []
    for step in range(1, cfg.steps + 1):
        raw_batch = next(dl_iter)
        if "index" in raw_batch:
            sampled_indices.extend(to_numpy(raw_batch["index"]).reshape(-1).astype(int).tolist())
        if "episode_index" in raw_batch:
            sampled_episodes.extend(to_numpy(raw_batch["episode_index"]).reshape(-1).astype(int).tolist())
        if "action" in raw_batch:
            sampled_gripper.extend(to_numpy(raw_batch["action"])[:, 0, 6].reshape(-1).astype(float).tolist())
        t_pre = time.perf_counter()
        raw_batch = apply_image_transforms(raw_batch, raw_cfg)
        batch = preprocessor(raw_batch)
        t_forward = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=bool(cfg.policy.use_amp)):
            if use_weighted_loss:
                loss, out_dict = smolvla_weighted_action_loss(
                    policy,
                    batch,
                    tcp_weight=tcp_loss_weight,
                    gripper_weight=gripper_loss_weight,
                )
            else:
                loss, out_dict = policy.forward(batch)
        t_backward = time.perf_counter()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
        t_optim = time.perf_counter()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        t_done = time.perf_counter()
        metrics.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "preprocess_latency_s": t_forward - t_pre,
                "forward_latency_s": t_backward - t_forward,
                "backward_latency_s": t_optim - t_backward,
                "optimizer_latency_s": t_done - t_optim,
                "output_dict": jsonable(out_dict),
            }
        )
        print(f"step={step} loss={metrics[-1]['loss']:.6f} grad_norm={metrics[-1]['grad_norm']:.4f}")

    after = dict(policy.named_parameters())[param_name].detach().float().cpu()
    diff = (after - before).abs()
    weight_report = {
        "parameter_name": param_name,
        "checksum_before": tensor_checksum(before),
        "checksum_after": tensor_checksum(after),
        "norm_before": float(before.norm()),
        "norm_after": float(after.norm()),
        "max_abs_difference": float(diff.max()),
        "updated": bool(float(diff.max()) > 0.0),
    }
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, cfg.steps)
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=cfg.steps,
        cfg=cfg,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    update_last_checkpoint(checkpoint_dir)
    summary.update(
        {
            "mode": training_mode_label(cfg),
            "steps": cfg.steps,
            "elapsed_s": time.perf_counter() - start_all,
            "checkpoint_dir": str(checkpoint_dir),
            "pretrained_model_dir": str(checkpoint_dir / "pretrained_model"),
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "weight_update": weight_report,
        }
    )
    (cfg.output_dir / "train_metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    (cfg.output_dir / "train_summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    if sampler_report is not None:
        sampled_indices_np = np.asarray(sampled_indices, dtype=np.int64)
        sampled_gripper_np = np.asarray(sampled_gripper, dtype=np.float32)
        sampled_episode_np = np.asarray(sampled_episodes, dtype=np.int64)
        sampled_transition = (
            transition_mask_for_index[sampled_indices_np]
            if transition_mask_for_index is not None and len(sampled_indices_np)
            else np.asarray([], dtype=bool)
        )
        episode_counts = {
            int(ep): int(np.sum(sampled_episode_np == ep)) for ep in sorted(set(sampled_episode_np.tolist()))
        }
        sampled_report = {
            "samples": int(len(sampled_indices_np)),
            "transition_ratio": float(np.mean(sampled_transition)) if len(sampled_transition) else 0.0,
            "open_ratio": float(np.mean(sampled_gripper_np == 0.0)) if len(sampled_gripper_np) else 0.0,
            "closed_ratio": float(np.mean(sampled_gripper_np == 1.0)) if len(sampled_gripper_np) else 0.0,
            "episode_counts": episode_counts,
        }
        write_gripper_sampler_report(reports / "gripper_sampler_report.md", sampler_report, sampled_report)
        (cfg.output_dir / "gripper_sampler_summary.json").write_text(
            json.dumps({"raw": jsonable(sampler_report), "sampled": sampled_report}, indent=2),
            encoding="utf-8",
        )
    weight_report_text = (
        "# Weight Update Verification\n\n"
        f"- Parameter: `{param_name}`\n"
        f"- Checksum before: `{weight_report['checksum_before']}`\n"
        f"- Checksum after: `{weight_report['checksum_after']}`\n"
        f"- Norm before: `{weight_report['norm_before']}`\n"
        f"- Norm after: `{weight_report['norm_after']}`\n"
        f"- Max absolute difference: `{weight_report['max_abs_difference']}`\n"
        f"- Updated: `{weight_report['updated']}`\n"
    )
    (cfg.output_dir / "weight_update_verification.md").write_text(weight_report_text, encoding="utf-8")
    if write_shared_reports:
        (reports / "weight_update_verification.md").write_text(weight_report_text, encoding="utf-8")
    print(json.dumps(jsonable(summary), indent=2)[:4000])
    return 0 if weight_report["updated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
