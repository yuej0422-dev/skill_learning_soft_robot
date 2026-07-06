from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from soft_vla.config import load_yaml


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.tolist() if value.numel() != 1 else value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def load_training_yaml(project_root: Path, config_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(project_root / config_path)
    ds_cfg = load_yaml(project_root / cfg["dataset"]["config"])["dataset"]
    cfg["_dataset_resolved"] = ds_cfg
    return cfg


def build_lerobot_train_config(project_root: Path, config: dict[str, Any]):
    from lerobot.configs.default import DatasetConfig, PeftConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig

    policy_cfg = config["policy"]
    dataset_cfg = config["_dataset_resolved"]
    root = Path(config["dataset"].get("root") or dataset_cfg["root"])
    if not root.is_absolute():
        root = project_root / root
    episodes = config["dataset"].get("episodes", dataset_cfg.get("episodes"))

    pcfg = PreTrainedConfig.from_pretrained(policy_cfg.get("pretrained_path", "lerobot/smolvla_base"))
    pcfg.input_features = {}
    pcfg.output_features = {}
    pcfg.pretrained_path = policy_cfg.get("pretrained_path", "lerobot/smolvla_base")
    pcfg.device = policy_cfg.get("device", "cuda")
    pcfg.use_amp = bool(policy_cfg.get("use_amp", True))
    pcfg.freeze_vision_encoder = bool(policy_cfg.get("freeze_vision_encoder", True))
    pcfg.train_expert_only = bool(policy_cfg.get("train_expert_only", True))
    pcfg.train_state_proj = bool(policy_cfg.get("train_state_proj", True))
    pcfg.push_to_hub = bool(policy_cfg.get("push_to_hub", False))
    pcfg.use_peft = bool(policy_cfg.get("use_peft", False))
    if "chunk_size" in policy_cfg:
        pcfg.chunk_size = int(policy_cfg["chunk_size"])
    if "n_action_steps" in policy_cfg:
        pcfg.n_action_steps = int(policy_cfg["n_action_steps"])

    peft_cfg = None
    if pcfg.use_peft:
        peft_cfg = PeftConfig(
            r=int(policy_cfg.get("lora_rank", 8)),
        )

    output_dir = Path(policy_cfg.get("checkpoint_dir", "outputs/smolvla_expert_smoke"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset_cfg.get("repo_id") or "local/synthetic_soft_robot_vla",
            root=str(root),
            episodes=episodes if isinstance(episodes, list) else None,
            use_imagenet_stats=False,
            video_backend=str(dataset_cfg.get("video_backend", config["dataset"].get("video_backend", "torchcodec"))),
        ),
        policy=pcfg,
        output_dir=output_dir,
        batch_size=int(policy_cfg.get("batch_size", 1)),
        steps=int(policy_cfg.get("steps", 20)),
        num_workers=int(policy_cfg.get("num_workers", 0)),
        eval_freq=int(policy_cfg.get("eval_freq", 0)),
        save_freq=int(policy_cfg.get("save_freq", policy_cfg.get("steps", 20))),
        log_freq=int(policy_cfg.get("log_freq", 1)),
        save_checkpoint=True,
        peft=peft_cfg,
    )
    return cfg


def parameter_summary(policy: torch.nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    names = [name for name, p in policy.named_parameters() if p.requires_grad]
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
        "trainable_name_count": len(names),
        "trainable_name_sample": names[:80],
    }


def first_trainable_parameter(policy: torch.nn.Module):
    for name, param in policy.named_parameters():
        if param.requires_grad:
            return name, param
    raise RuntimeError("No trainable parameter found.")
