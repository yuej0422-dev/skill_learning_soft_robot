from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def _as_tensor(values: Iterable[float] | np.ndarray | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().clone().to(dtype=dtype)
    return torch.as_tensor(np.asarray(values), dtype=dtype)


def build_mlp(layer_sizes: list[int], activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if i != len(layer_sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class FeedforwardPressurePolicy(nn.Module):
    """Map raw target state to feedforward pressure.

    The module stores LeRobot state statistics and normalizes inside forward(),
    so callers can pass raw target states at inference time.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        state_mean: Iterable[float] | np.ndarray | torch.Tensor,
        state_std: Iterable[float] | np.ndarray | torch.Tensor,
        state_indices: Iterable[int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer_sizes = [int(v) for v in layer_sizes]
        self.state_indices = [int(v) for v in state_indices]
        self.eps = float(eps)

        selected_mean = _as_tensor(state_mean)[self.state_indices]
        selected_std = _as_tensor(state_std)[self.state_indices].clamp_min(self.eps)
        self.register_buffer("state_mean", selected_mean)
        self.register_buffer("state_std", selected_std)
        self.net = build_mlp(self.layer_sizes)

    @property
    def input_dim(self) -> int:
        return len(self.state_indices)

    @property
    def output_dim(self) -> int:
        return self.layer_sizes[-1]

    def normalize_state(self, raw_state: torch.Tensor) -> torch.Tensor:
        if raw_state.ndim == 1:
            raw_state = raw_state.unsqueeze(0)
        if raw_state.shape[-1] == len(self.state_indices):
            selected = raw_state
        elif raw_state.shape[-1] > max(self.state_indices):
            idx = torch.as_tensor(self.state_indices, device=raw_state.device)
            selected = raw_state.index_select(dim=-1, index=idx)
        else:
            raise ValueError(
                f"Expected state dimension {len(self.state_indices)} or at least "
                f"{max(self.state_indices) + 1}, got {raw_state.shape[-1]}."
            )
        return (selected - self.state_mean.to(raw_state.device)) / self.state_std.to(raw_state.device)

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize_state(raw_state.float()))

    @torch.no_grad()
    def predict_pressure(self, raw_state: Iterable[float] | np.ndarray | torch.Tensor) -> np.ndarray:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        input_was_1d = raw_state.ndim == 1 if isinstance(raw_state, torch.Tensor) else np.asarray(raw_state).ndim == 1
        state = _as_tensor(raw_state).to(device)
        pressure = self(state).detach().cpu().numpy()
        if was_training:
            self.train()
        return pressure[0] if input_was_1d else pressure


@dataclass(frozen=True)
class PolicyMetadata:
    state_indices: list[int]
    state_names: list[str]
    pressure_columns: list[str]
    dataset_root: str
    input_key: str = "observation.state"
    target_key: str = "raw_pressure"
