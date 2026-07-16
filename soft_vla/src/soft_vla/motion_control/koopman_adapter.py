from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KoopmanAdapterConfig:
    checkpoint: str | Path
    device: str = "cpu"


class KoopmanAdapter:
    def __init__(self, config: KoopmanAdapterConfig) -> None:
        self.config = config
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from motion_control_training.koopman.model import KoopmanNetwork
        except ImportError as exc:  # pragma: no cover - optional runtime dep
            raise RuntimeError("torch and motion_control_training.koopman are required for KoopmanAdapter") from exc

        ckpt = torch.load(self.config.checkpoint, map_location=self.config.device)
        self.metadata = ckpt.get("metadata", {})
        self.state_mean = np.asarray(self.metadata.get("state_mean", np.zeros(12)), dtype=np.float32)
        self.state_std = np.maximum(np.asarray(self.metadata.get("state_std", np.ones(12)), dtype=np.float32), 1e-6)
        self.model = KoopmanNetwork(
            encode_layers=[int(v) for v in ckpt["encode_layers"]],
            n_koopman=int(ckpt["n_koopman"]),
            u_dim=int(ckpt["u_dim"]),
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.config.device)
        self.model.eval()
        self.torch = torch
        self.A_lift = self.model.form_A_from_eigenvalues().detach().cpu().numpy().astype(np.float64)
        self.B = self.model.B.detach().cpu().numpy().astype(np.float64)
        self.n_koopman = int(ckpt["n_koopman"])
        self.u_dim = int(ckpt["u_dim"])

    def normalize_state(self, state12: np.ndarray) -> np.ndarray:
        state = np.asarray(state12, dtype=np.float32).reshape(12)
        return ((state - self.state_mean) / self.state_std).astype(np.float32)

    def lift(self, state12: np.ndarray) -> np.ndarray:
        norm = self.normalize_state(state12)
        with self.torch.no_grad():
            x = self.torch.as_tensor(norm, dtype=self.torch.float32, device=self.config.device).unsqueeze(0)
            z = self.model.encode(x).detach().cpu().numpy()[0]
        return z.astype(np.float64)

    def tracking_error(self, current_state12: np.ndarray, reference_state12: np.ndarray) -> np.ndarray:
        return self.lift(current_state12) - self.lift(reference_state12)

    def output_matrix(self, ny: int) -> np.ndarray:
        if ny < 0 or ny > 12:
            raise ValueError("ny must be between 0 and 12")
        c_full = np.concatenate([np.eye(12), np.zeros((12, self.n_koopman - 12))], axis=1)
        return c_full[:ny].astype(np.float64)

