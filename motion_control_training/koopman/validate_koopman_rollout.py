from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .model import KoopmanNetwork
    from .train_koopman_lerobot import (
        DEFAULT_DATASET_ROOT,
        load_episode_arrays,
        load_lerobot_state_stats,
        parse_int_list,
        resolve_device,
        upsample_episode_arrays,
    )
except ImportError:  # pragma: no cover - direct script execution
    from model import KoopmanNetwork
    from train_koopman_lerobot import (
        DEFAULT_DATASET_ROOT,
        load_episode_arrays,
        load_lerobot_state_stats,
        parse_int_list,
        resolve_device,
        upsample_episode_arrays,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[KoopmanNetwork, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = KoopmanNetwork(
        encode_layers=[int(v) for v in checkpoint["encode_layers"]],
        n_koopman=int(checkpoint["n_koopman"]),
        u_dim=int(checkpoint["u_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def choose_episodes(args: argparse.Namespace, checkpoint: dict, available: list[int]) -> list[int]:
    if args.episodes:
        selected = parse_int_list(args.episodes)
    else:
        metadata = checkpoint.get("metadata", {})
        if args.split == "val":
            selected = list(metadata.get("val_episodes", []))
        elif args.split == "train":
            selected = list(metadata.get("train_episodes", []))
        else:
            selected = available
    selected = [int(ep) for ep in selected if int(ep) in set(available)]
    if args.max_episodes > 0:
        selected = selected[: args.max_episodes]
    if not selected:
        raise ValueError("No episodes selected for validation.")
    return selected


@torch.no_grad()
def one_step_metrics(
    model: KoopmanNetwork,
    states_norm: np.ndarray,
    pressures: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.as_tensor(states_norm[:-1], dtype=torch.float32, device=device)
    u = torch.as_tensor(pressures[:-1], dtype=torch.float32, device=device)
    target = states_norm[1:]
    lifted = model.encode(x)
    pred_norm = model(lifted, u)[:, : states_norm.shape[1]].detach().cpu().numpy()
    err_norm = pred_norm - target
    err_raw = err_norm * state_std
    return err_norm, err_raw, pred_norm


@torch.no_grad()
def rollout_metrics(
    model: KoopmanNetwork,
    states_norm: np.ndarray,
    pressures: np.ndarray,
    state_std: np.ndarray,
    rollout_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = min(int(rollout_steps), len(states_norm) - 1)
    if horizon <= 0:
        empty = np.empty((0, states_norm.shape[1]), dtype=np.float32)
        return empty, empty, empty

    current = torch.as_tensor(states_norm[0:1], dtype=torch.float32, device=device)
    lifted = model.encode(current)
    preds: list[np.ndarray] = []
    for step in range(horizon):
        control = torch.as_tensor(pressures[step : step + 1], dtype=torch.float32, device=device)
        lifted = model(lifted, control)
        pred_state = lifted[:, : states_norm.shape[1]]
        preds.append(pred_state.detach().cpu().numpy()[0])
        lifted = torch.cat([pred_state, model.encode_only(pred_state)], dim=-1)

    pred_norm = np.stack(preds, axis=0)
    target = states_norm[1 : horizon + 1]
    err_norm = pred_norm - target
    err_raw = err_norm * state_std
    return err_norm, err_raw, pred_norm


def rmse(values: np.ndarray) -> tuple[float, list[float]]:
    if values.size == 0:
        return float("nan"), []
    per_dim = np.sqrt(np.mean(np.square(values), axis=0))
    return float(np.mean(per_dim)), [float(v) for v in per_dim]


def validate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint.resolve(), device)
    metadata = checkpoint.get("metadata", {})

    dataset_root = args.dataset_root
    if dataset_root is None:
        configured_root = checkpoint.get("config", {}).get("dataset_root")
        dataset_root = Path(configured_root) if configured_root else DEFAULT_DATASET_ROOT
    dataset_root = dataset_root.resolve()

    state_indices = [int(v) for v in metadata.get("state_indices", parse_int_list(args.state_indices))]
    pressure_indices = [int(v) for v in metadata.get("pressure_indices", parse_int_list(args.pressure_indices))]
    state_mean = np.asarray(metadata.get("state_mean"), dtype=np.float32)
    state_std = np.asarray(metadata.get("state_std"), dtype=np.float32)
    if state_mean.size == 0 or state_std.size == 0:
        full_mean, full_std, _ = load_lerobot_state_stats(dataset_root)
        state_mean = full_mean[state_indices]
        state_std = full_std[state_indices]
    state_std = np.maximum(state_std, args.norm_eps)
    if args.upsample_factor is None:
        upsample_factor = int(metadata.get("upsample_factor", 1))
    else:
        upsample_factor = int(args.upsample_factor)

    episodes = load_episode_arrays(dataset_root, state_indices, pressure_indices)
    selected_episodes = choose_episodes(args, checkpoint, sorted(episodes))

    one_step_norm_errors: list[np.ndarray] = []
    one_step_raw_errors: list[np.ndarray] = []
    rollout_norm_errors: list[np.ndarray] = []
    rollout_raw_errors: list[np.ndarray] = []
    saved_predictions: dict[str, np.ndarray] = {}
    episode_summaries: list[dict] = []

    for episode in selected_episodes:
        states_raw, pressures = episodes[episode]
        states_raw, pressures = upsample_episode_arrays(states_raw, pressures, upsample_factor)
        states_norm = (states_raw - state_mean) / state_std

        err_norm, err_raw, one_step_pred = one_step_metrics(model, states_norm, pressures, state_std, device)
        ro_err_norm, ro_err_raw, rollout_pred = rollout_metrics(
            model,
            states_norm,
            pressures,
            state_std,
            args.rollout_steps,
            device,
        )
        one_step_norm_errors.append(err_norm)
        one_step_raw_errors.append(err_raw)
        rollout_norm_errors.append(ro_err_norm)
        rollout_raw_errors.append(ro_err_raw)

        one_step_rmse_mean, _ = rmse(err_raw)
        rollout_rmse_mean, _ = rmse(ro_err_raw)
        episode_summaries.append(
            {
                "episode": int(episode),
                "frames": int(len(states_raw)),
                "one_step_raw_rmse_mean": one_step_rmse_mean,
                "rollout_raw_rmse_mean": rollout_rmse_mean,
            }
        )

        if args.save_npz:
            prefix = f"episode_{episode:06d}"
            saved_predictions[f"{prefix}_one_step_pred_norm"] = one_step_pred
            saved_predictions[f"{prefix}_rollout_pred_norm"] = rollout_pred
            saved_predictions[f"{prefix}_target_norm"] = states_norm
            saved_predictions[f"{prefix}_pressure"] = pressures

    one_step_norm = np.concatenate(one_step_norm_errors, axis=0)
    one_step_raw = np.concatenate(one_step_raw_errors, axis=0)
    rollout_norm = np.concatenate(rollout_norm_errors, axis=0)
    rollout_raw = np.concatenate(rollout_raw_errors, axis=0)

    one_step_norm_mean, one_step_norm_per_dim = rmse(one_step_norm)
    one_step_raw_mean, one_step_raw_per_dim = rmse(one_step_raw)
    rollout_norm_mean, rollout_norm_per_dim = rmse(rollout_norm)
    rollout_raw_mean, rollout_raw_per_dim = rmse(rollout_raw)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(dataset_root),
        "device": str(device),
        "split": args.split,
        "episodes": selected_episodes,
        "state_indices": state_indices,
        "pressure_indices": pressure_indices,
        "rollout_steps": int(args.rollout_steps),
        "upsample_factor": int(upsample_factor),
        "upsample_method": "state_linear_interpolation_pressure_zero_order_hold",
        "one_step": {
            "normalized_rmse_mean": one_step_norm_mean,
            "normalized_rmse_per_dim": one_step_norm_per_dim,
            "raw_rmse_mean": one_step_raw_mean,
            "raw_rmse_per_dim": one_step_raw_per_dim,
        },
        "rollout": {
            "normalized_rmse_mean": rollout_norm_mean,
            "normalized_rmse_per_dim": rollout_norm_per_dim,
            "raw_rmse_mean": rollout_raw_mean,
            "raw_rmse_per_dim": rollout_raw_per_dim,
        },
        "episode_summaries": episode_summaries,
    }

    output_json = args.output_json
    if output_json is None:
        output_json = args.checkpoint.resolve().parent / "validation_rollout.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_npz:
        npz_path = output_json.with_suffix(".npz")
        np.savez_compressed(npz_path, **saved_predictions)
        summary["prediction_npz"] = str(npz_path)
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"episodes={selected_episodes}")
    print(
        f"one_step raw_rmse_mean={one_step_raw_mean:.8f} "
        f"normalized_rmse_mean={one_step_norm_mean:.8f}"
    )
    print(
        f"rollout raw_rmse_mean={rollout_raw_mean:.8f} "
        f"normalized_rmse_mean={rollout_norm_mean:.8f}"
    )
    print(f"Saved validation summary to {output_json}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained Koopman checkpoint with one-step and rollout metrics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--state-indices", type=str, default="0:12")
    parser.add_argument("--pressure-indices", type=str, default="0:12")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"])
    parser.add_argument("--episodes", type=str, default="", help="Optional comma/slice episode ids, e.g. 0,1,2.")
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=50)
    parser.add_argument("--upsample-factor", type=int, default=None, help="Defaults to checkpoint metadata; older checkpoints use 1.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-npz", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    validate(args)


if __name__ == "__main__":
    main()
