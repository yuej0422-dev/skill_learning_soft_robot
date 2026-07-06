from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .model import FeedforwardPressurePolicy
except ImportError:  # pragma: no cover - direct script execution
    from model import FeedforwardPressurePolicy


def load_policy(checkpoint_path: str | Path, device: str = "cpu") -> FeedforwardPressurePolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    layer_sizes = [int(v) for v in checkpoint["layer_sizes"]]
    state_indices = [int(v) for v in checkpoint["state_indices"]]

    full_state_dim = max(state_indices) + 1
    state_mean = np.zeros(full_state_dim, dtype=np.float32)
    state_std = np.ones(full_state_dim, dtype=np.float32)
    saved_mean = checkpoint["state_mean"].detach().cpu().numpy()
    saved_std = checkpoint["state_std"].detach().cpu().numpy()
    state_mean[state_indices] = saved_mean
    state_std[state_indices] = saved_std

    policy = FeedforwardPressurePolicy(
        layer_sizes=layer_sizes,
        state_mean=state_mean,
        state_std=state_std,
        state_indices=state_indices,
    )
    policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
    policy.to(device)
    policy.eval()
    return policy


def parse_state(spec: str) -> np.ndarray:
    values = [float(item) for item in spec.replace(";", ",").split(",") if item.strip()]
    if not values:
        raise ValueError("State is empty.")
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer feedforward pressure from raw target state.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=str, default=None, help="Comma-separated raw state values.")
    parser.add_argument("--state-npy", type=Path, default=None, help="Optional .npy file with shape (D,) or (N,D).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--clip-min", type=float, default=None, help="Optional lower bound for printed pressure.")
    parser.add_argument("--clip-max", type=float, default=None, help="Optional upper bound for printed pressure.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain numpy formatting.")
    args = parser.parse_args()

    if (args.state is None) == (args.state_npy is None):
        raise SystemExit("Provide exactly one of --state or --state-npy.")

    raw_state = parse_state(args.state) if args.state is not None else np.load(args.state_npy).astype(np.float32)
    policy = load_policy(args.checkpoint, device=args.device)
    pressure = policy.predict_pressure(raw_state)
    if args.clip_min is not None or args.clip_max is not None:
        pressure = np.clip(pressure, args.clip_min, args.clip_max)

    if args.json:
        print(json.dumps({"pressure": np.asarray(pressure).tolist()}, ensure_ascii=False))
    else:
        print(np.asarray(pressure))


if __name__ == "__main__":
    main()
