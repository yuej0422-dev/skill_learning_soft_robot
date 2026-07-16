from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np


class FeedforwardPolicy(Protocol):
    def predict(
        self,
        *,
        current_state12: np.ndarray,
        reference_state12: np.ndarray,
        delta_tcp6: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FeedforwardPressureConfig:
    checkpoint: str | Path
    device: str = "cpu"
    input_mode: str = "target_state"


class ZeroFeedforwardPolicy:
    def predict(
        self,
        *,
        current_state12: np.ndarray,
        reference_state12: np.ndarray,
        delta_tcp6: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(12, dtype=np.float32)


class FeedforwardPressureMLPAdapter:
    def __init__(self, config: FeedforwardPressureConfig) -> None:
        self.config = config
        self.policy = _load_pressure_mlp(config.checkpoint, device=config.device)
        self.input_mode = str(config.input_mode)

    def predict(
        self,
        *,
        current_state12: np.ndarray,
        reference_state12: np.ndarray,
        delta_tcp6: np.ndarray,
    ) -> np.ndarray:
        if self.input_mode == "target_state":
            raw = np.asarray(reference_state12, dtype=np.float32)
        elif self.input_mode == "observation_state":
            raw = np.asarray(current_state12, dtype=np.float32)
        else:
            raise ValueError(f"unsupported feedforward input_mode: {self.input_mode}")
        out = self.policy.predict_pressure(raw)
        return np.clip(np.asarray(out, dtype=np.float32).reshape(12), 0.0, 1.0)


@dataclass(frozen=True)
class AwacFeedforwardConfig:
    checkpoint: str | Path
    device: str = "cpu"


class AwacFeedforwardAdapter:
    def __init__(self, config: AwacFeedforwardConfig) -> None:
        self.config = config
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - optional runtime dep
            raise RuntimeError("torch is required for AWAC feedforward inference") from exc

        checkpoint = torch.load(self.config.checkpoint, map_location=self.config.device)
        metadata = checkpoint["metadata"]
        state_dim = 2 * len(metadata["state_indices"])
        action_low = np.asarray(metadata["action_low"], dtype=np.float32)
        action_high = np.asarray(metadata["action_high"], dtype=np.float32)
        hidden_sizes = [int(v) for v in str(checkpoint["config"]["hidden_sizes"]).split(",") if v]

        class Actor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                target_dim = state_dim // 2
                sizes = [target_dim] + hidden_sizes + [len(action_low)]
                layers: list[nn.Module] = []
                for i, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
                    layers.append(nn.Linear(in_dim, out_dim))
                    if i != len(sizes) - 2:
                        layers.append(nn.ReLU())
                self.net = nn.Sequential(*layers)
                self.state_dim = state_dim
                self.log_std = nn.Parameter(torch.zeros(len(action_low), dtype=torch.float32))
                self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
                self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

            def forward(self, state):
                target = state[..., self.state_dim // 2 :]
                action = self.net(target)
                return torch.max(torch.min(action, self.action_high.to(action.device)), self.action_low.to(action.device))

        actor = Actor().to(self.config.device)
        actor.load_state_dict(checkpoint["trainer_state_dict"]["actor"], strict=True)
        actor.eval()
        self.torch = torch
        self.actor = actor
        if "state_mean" in metadata and "state_std" in metadata:
            self.state_mean = np.asarray(metadata["state_mean"], dtype=np.float32)
            self.state_std = np.asarray(metadata["state_std"], dtype=np.float32)
        elif "state_mean" in checkpoint and "state_std" in checkpoint:
            self.state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
            self.state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
        else:
            self.state_mean, self.state_std = _load_state_stats_from_dataset(checkpoint)
        if self.state_mean.shape != (12,):
            self.state_mean, self.state_std = _load_state_stats_from_dataset(checkpoint)
        self.state_std = np.maximum(self.state_std, 1e-6)

    def predict(
        self,
        *,
        current_state12: np.ndarray,
        reference_state12: np.ndarray,
        delta_tcp6: np.ndarray,
    ) -> np.ndarray:
        current = (np.asarray(current_state12, dtype=np.float32) - self.state_mean) / self.state_std
        target = (np.asarray(reference_state12, dtype=np.float32) - self.state_mean) / self.state_std
        obs = np.concatenate([current, target]).astype(np.float32)
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(obs, dtype=self.torch.float32, device=self.config.device).unsqueeze(0)
            action = self.actor(tensor).detach().cpu().numpy()[0]
        return np.clip(action.astype(np.float32), 0.0, 1.0)


def _load_pressure_mlp(checkpoint: str | Path, *, device: str):
    try:
        from motion_control_training.feedforward_pressure.infer_pressure import load_policy
    except ImportError as exc:  # pragma: no cover - depends on repo root/env
        raise RuntimeError("cannot import feedforward pressure loader") from exc
    return load_policy(checkpoint, device=device)


def _load_state_stats_from_dataset(checkpoint: dict) -> tuple[np.ndarray, np.ndarray]:
    dataset_root = checkpoint.get("config", {}).get("dataset_root")
    state_indices = checkpoint.get("metadata", {}).get("state_indices", list(range(12)))
    if dataset_root:
        stats_path = Path(dataset_root) / "meta/stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            mean = np.asarray(stats["observation.state"]["mean"], dtype=np.float32)[state_indices]
            std = np.asarray(stats["observation.state"]["std"], dtype=np.float32)[state_indices]
            return mean.astype(np.float32), std.astype(np.float32)
    return np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)
