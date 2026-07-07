from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_control_training.koopman.train_koopman_lerobot import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    load_episode_arrays,
    load_lerobot_state_stats,
    load_pressure_metadata,
    parse_hidden_sizes,
    parse_int_list,
    resolve_device,
    split_episodes,
)

try:
    from .model_fullA_history import (
        FullAHistoryKoopmanNetwork,
        FullAHistoryLossWeights,
        define_fullA_history_loss,
    )
except ImportError:  # pragma: no cover
    from model_fullA_history import (
        FullAHistoryKoopmanNetwork,
        FullAHistoryLossWeights,
        define_fullA_history_loss,
    )


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dir(output_root: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def context_at(
    states: np.ndarray,
    pressures: np.ndarray,
    t: int,
    history_steps: int,
) -> np.ndarray:
    state_history = states[t - history_steps + 1 : t + 1]
    action_history = pressures[t - history_steps : t]
    if state_history.shape[0] != history_steps or action_history.shape[0] != history_steps:
        raise IndexError(f"Invalid context at t={t} with history_steps={history_steps}")
    return np.concatenate([state_history.reshape(-1), action_history.reshape(-1)], axis=0).astype(np.float32)


def build_history_koopman_buffer(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: Sequence[int],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    history_steps: int,
    ksteps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    contexts: list[np.ndarray] = []
    current_states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    skipped_short = 0
    original_frames = 0
    processed_frames = 0

    for episode in episode_ids:
        states_raw, pressures = episodes[int(episode)]
        original_frames += int(len(states_raw))
        processed_frames += int(len(states_raw))
        if len(states_raw) < history_steps + ksteps + 1:
            skipped_short += 1
            continue

        norm_states = ((states_raw - state_mean) / state_std).astype(np.float32)
        pressures = pressures.astype(np.float32)
        for t in range(history_steps, len(norm_states) - ksteps):
            context_sequence = np.stack(
                [context_at(norm_states, pressures, t + offset, history_steps) for offset in range(ksteps + 1)],
                axis=0,
            )
            current_state_sequence = norm_states[t : t + ksteps + 1]
            control_sequence = pressures[t : t + ksteps]
            state_target_sequence = norm_states[t + 1 : t + ksteps + 1]
            contexts.append(context_sequence)
            current_states.append(current_state_sequence)
            controls.append(control_sequence)
            targets.append(state_target_sequence)

    if not contexts:
        raise ValueError(
            "No windows built. Need episode length >= history_steps + ksteps + 1 "
            f"({history_steps + ksteps + 1})."
        )

    context_array = np.stack(contexts, axis=0).astype(np.float32)
    state_array = np.stack(current_states, axis=0).astype(np.float32)
    control_array = np.stack(controls, axis=0).astype(np.float32)
    target_array = np.stack(targets, axis=0).astype(np.float32)
    stats = {
        "episodes": len(episode_ids),
        "skipped_short_episodes": skipped_short,
        "original_frames": original_frames,
        "processed_frames": processed_frames,
        "windows": int(context_array.shape[0]),
        "history_steps": int(history_steps),
        "ksteps": int(ksteps),
        "upsample_factor": 1,
    }
    return context_array, state_array, control_array, target_array, stats


def tensor_components_to_weighted_float(components: dict[str, torch.Tensor], batch_size: int) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) * batch_size for key, value in components.items()}


def average_components(rows: list[dict[str, float]], total_count: int) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / max(total_count, 1) for key in rows[0]}


@torch.no_grad()
def evaluate(
    model: FullAHistoryKoopmanNetwork,
    loader: DataLoader,
    mse_loss: nn.Module,
    device: torch.device,
    gamma: float,
    weights: FullAHistoryLossWeights,
    spectral_radius_limit: float,
    target_std: float,
    svd_min_singular_value: float,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    total_count = 0
    for contexts, states, controls, targets in loader:
        contexts = contexts.to(device)
        states = states.to(device)
        controls = controls.to(device)
        targets = targets.to(device)
        _, components = define_fullA_history_loss(
            contexts,
            states,
            controls,
            targets,
            model,
            mse_loss,
            gamma,
            weights,
            spectral_radius_limit,
            target_std,
            svd_min_singular_value,
        )
        batch_size = contexts.shape[0]
        total_count += batch_size
        rows.append(tensor_components_to_weighted_float(components, batch_size))
    return average_components(rows, total_count)


def save_checkpoint(
    path: Path,
    model: FullAHistoryKoopmanNetwork,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
    metadata: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "context_dim": model.context_dim,
            "n_state": model.n_state,
            "u_dim": model.u_dim,
            "encode_dim": model.encode_dim,
            "hidden_sizes": model.hidden_sizes,
            "n_koopman": model.n_koopman,
            "config": config,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[FullAHistoryKoopmanNetwork, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FullAHistoryKoopmanNetwork(
        context_dim=int(checkpoint["context_dim"]),
        n_state=int(checkpoint["n_state"]),
        u_dim=int(checkpoint["u_dim"]),
        encode_dim=int(checkpoint["encode_dim"]),
        hidden_sizes=[int(v) for v in checkpoint["hidden_sizes"]],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def init_wandb(args: argparse.Namespace, output_dir: Path, config: dict, metadata: dict):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("wandb is not installed. Install wandb or omit --wandb.") from exc
    tags = [tag for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or output_dir.name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        tags=tags,
        config={**config, "metadata": metadata},
    )


def gradient_diagnostics(model: FullAHistoryKoopmanNetwork) -> dict[str, float]:
    encoder_grad_sq = 0.0
    for parameter in model.encoder.parameters():
        if parameter.grad is not None:
            encoder_grad_sq += float(parameter.grad.detach().pow(2).sum().cpu())
    return {
        "A_grad_norm": float(model.A.grad.detach().norm().cpu()) if model.A.grad is not None else 0.0,
        "B_grad_norm": float(model.B.grad.detach().norm().cpu()) if model.B.grad is not None else 0.0,
        "bias_grad_norm": float(model.bias.grad.detach().norm().cpu()) if model.bias.grad is not None else 0.0,
        "encoder_grad_norm": float(encoder_grad_sq**0.5),
        "A_latent_to_state_grad_norm": (
            float(model.A.grad[model.n_state :, : model.n_state].detach().norm().cpu())
            if model.A.grad is not None
            else 0.0
        ),
    }


def train(args: argparse.Namespace) -> Path:
    if args.upsample_factor != 1:
        raise ValueError("fullA_history_v2 uses original 10 Hz data only. Use --upsample-factor 1.")
    if args.source_hz != 10 or args.target_hz != 10:
        raise ValueError("fullA_history_v2 requires --source-hz 10 and --target-hz 10.")

    set_seed(args.seed)
    dataset_root = args.dataset_root.resolve()
    full_state_mean, full_state_std, state_names = load_lerobot_state_stats(dataset_root)
    state_indices = parse_int_list(args.state_indices, total_dim=len(full_state_mean))
    pressure_indices = parse_int_list(args.pressure_indices)
    state_mean = full_state_mean[state_indices]
    state_std = np.maximum(full_state_std[state_indices], args.norm_eps)
    n_state = len(state_indices)
    u_dim = len(pressure_indices)
    context_dim = args.history_steps * n_state + args.history_steps * u_dim

    pressure_columns, _ = load_pressure_metadata(dataset_root)
    episodes = load_episode_arrays(dataset_root, state_indices, pressure_indices)
    train_episodes, val_episodes = split_episodes(sorted(episodes), args.val_ratio, args.seed)
    train_contexts, train_states, train_controls, train_targets, train_stats = build_history_koopman_buffer(
        episodes,
        train_episodes,
        state_mean,
        state_std,
        args.history_steps,
        args.ksteps,
    )
    val_contexts, val_states, val_controls, val_targets, val_stats = build_history_koopman_buffer(
        episodes,
        val_episodes,
        state_mean,
        state_std,
        args.history_steps,
        args.ksteps,
    )

    if args.max_train_windows > 0 and train_contexts.shape[0] > args.max_train_windows:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(train_contexts.shape[0], size=args.max_train_windows, replace=False)
        train_contexts, train_states, train_controls, train_targets = (
            train_contexts[keep],
            train_states[keep],
            train_controls[keep],
            train_targets[keep],
        )
        train_stats["windows_after_subsample"] = int(train_contexts.shape[0])
    if args.max_val_windows > 0 and val_contexts.shape[0] > args.max_val_windows:
        rng = np.random.default_rng(args.seed + 1)
        keep = rng.choice(val_contexts.shape[0], size=args.max_val_windows, replace=False)
        val_contexts, val_states, val_controls, val_targets = (
            val_contexts[keep],
            val_states[keep],
            val_controls[keep],
            val_targets[keep],
        )
        val_stats["windows_after_subsample"] = int(val_contexts.shape[0])

    device = torch.device(args.device)
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_contexts, dtype=torch.float32),
            torch.as_tensor(train_states, dtype=torch.float32),
            torch.as_tensor(train_controls, dtype=torch.float32),
            torch.as_tensor(train_targets, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(val_contexts, dtype=torch.float32),
            torch.as_tensor(val_states, dtype=torch.float32),
            torch.as_tensor(val_controls, dtype=torch.float32),
            torch.as_tensor(val_targets, dtype=torch.float32),
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )

    model = FullAHistoryKoopmanNetwork(
        context_dim=context_dim,
        n_state=n_state,
        u_dim=u_dim,
        encode_dim=args.encode_dim,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()
    weights = FullAHistoryLossWeights(
        koopman=args.koopman_lam,
        pred=args.pred_lam,
        stability=args.stability_lam,
        std=args.std_lam,
        identity=args.identity_lam,
        svd=args.svd_lam,
        augment=args.augment_lam,
    )

    output_dir = make_output_dir(args.output_root.resolve(), args.run_name)
    config = vars(args).copy()
    config["dataset_root"] = str(dataset_root)
    config["output_root"] = str(args.output_root.resolve())
    metadata = {
        "experiment": "fullA_history_v2",
        "state_key": "observation.state",
        "state_indices": state_indices,
        "state_names": [state_names[i] for i in state_indices],
        "state_normalization": "lerobot_observation_state_mean_std",
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "pressure_key": "raw_pressure",
        "pressure_indices": pressure_indices,
        "pressure_columns": [pressure_columns[i] for i in pressure_indices],
        "pressure_normalization": "none",
        "pressure_centering": "none",
        "source_hz": float(args.source_hz),
        "target_hz": float(args.target_hz),
        "upsample_factor": int(args.upsample_factor),
        "upsample_method": "none",
        "history_steps": int(args.history_steps),
        "rollout_steps": int(args.ksteps),
        "ksteps": int(args.ksteps),
        "context_dim": int(context_dim),
        "buffer_layout": "context_sequence[K+1], current_state_sequence[K+1], control_sequence[K], state_target_sequence[K]",
        "no_cross_episode_windows": True,
        "train_drop_last": bool(args.drop_last),
        "train_buffer": train_stats,
        "val_buffer": val_stats,
    }
    (output_dir / "config.json").write_text(
        json.dumps({"config": config, "metadata": metadata}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    wandb_run = init_wandb(args, output_dir, config, metadata)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_linear_loss",
        "train_pred_loss",
        "train_stability_loss",
        "train_std_loss",
        "train_identity_loss",
        "train_svd_loss",
        "train_augment_loss",
        "val_loss",
        "val_linear_loss",
        "val_pred_loss",
        "val_stability_loss",
        "val_std_loss",
        "val_identity_loss",
        "val_svd_loss",
        "val_augment_loss",
        "train_latent_std_min",
        "train_latent_std_mean",
        "train_latent_std_max",
        "val_latent_std_min",
        "val_latent_std_mean",
        "val_latent_std_max",
        "spectral_radius",
        "eig_abs_mean",
        "eig_abs_max",
        "A_norm",
        "B_norm",
        "A_latent_to_state_norm",
        "A_state_to_latent_norm",
        "A_grad_norm",
        "B_grad_norm",
        "bias_grad_norm",
        "encoder_grad_norm",
        "A_latent_to_state_grad_norm",
        "lr",
        "epoch_seconds",
    ]

    print(f"dataset_root={dataset_root}")
    print(
        f"device={device} train_windows={train_contexts.shape[0]} val_windows={val_contexts.shape[0]} "
        f"history_steps={args.history_steps} ksteps={args.ksteps} context_dim={context_dim}"
    )
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    metrics_path = output_dir / "metrics.csv"
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epoch = 0
    epochs_without_improvement = 0

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                start = time.perf_counter()
                completed_epoch = epoch
                model.train()
                train_rows: list[dict[str, float]] = []
                grad_rows: list[dict[str, float]] = []
                train_count = 0
                for contexts, states, controls, targets in train_loader:
                    contexts = contexts.to(device)
                    states = states.to(device)
                    controls = controls.to(device)
                    targets = targets.to(device)
                    loss, components = define_fullA_history_loss(
                        contexts,
                        states,
                        controls,
                        targets,
                        model,
                        mse_loss,
                        args.gamma,
                        weights,
                        args.spectral_radius_limit,
                        args.target_std,
                        args.svd_min_singular_value,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_rows.append(gradient_diagnostics(model))
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    batch_size = contexts.shape[0]
                    train_count += batch_size
                    train_rows.append(tensor_components_to_weighted_float(components, batch_size))

                train_metrics = average_components(train_rows, train_count)
                val_metrics = evaluate(
                    model,
                    val_loader,
                    mse_loss,
                    device,
                    args.gamma,
                    weights,
                    args.spectral_radius_limit,
                    args.target_std,
                    args.svd_min_singular_value,
                )
                grad_metrics = {key: float(np.mean([row[key] for row in grad_rows])) for key in grad_rows[0]}
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_linear_loss": train_metrics["linear_loss"],
                    "train_pred_loss": train_metrics["pred_loss"],
                    "train_stability_loss": train_metrics["stability_loss"],
                    "train_std_loss": train_metrics["std_loss"],
                    "train_identity_loss": train_metrics["identity_loss"],
                    "train_svd_loss": train_metrics["svd_loss"],
                    "train_augment_loss": train_metrics["augment_loss"],
                    "val_loss": val_metrics["loss"],
                    "val_linear_loss": val_metrics["linear_loss"],
                    "val_pred_loss": val_metrics["pred_loss"],
                    "val_stability_loss": val_metrics["stability_loss"],
                    "val_std_loss": val_metrics["std_loss"],
                    "val_identity_loss": val_metrics["identity_loss"],
                    "val_svd_loss": val_metrics["svd_loss"],
                    "val_augment_loss": val_metrics["augment_loss"],
                    "train_latent_std_min": train_metrics["latent_std_min"],
                    "train_latent_std_mean": train_metrics["latent_std_mean"],
                    "train_latent_std_max": train_metrics["latent_std_max"],
                    "val_latent_std_min": val_metrics["latent_std_min"],
                    "val_latent_std_mean": val_metrics["latent_std_mean"],
                    "val_latent_std_max": val_metrics["latent_std_max"],
                    "spectral_radius": val_metrics["spectral_radius"],
                    "eig_abs_mean": val_metrics["eig_abs_mean"],
                    "eig_abs_max": val_metrics["eig_abs_max"],
                    "A_norm": val_metrics["A_norm"],
                    "B_norm": val_metrics["B_norm"],
                    "A_latent_to_state_norm": val_metrics["A_latent_to_state_norm"],
                    "A_state_to_latent_norm": val_metrics["A_state_to_latent_norm"],
                    **grad_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - start,
                }
                writer.writerow(row)
                f.flush()
                if wandb_run is not None:
                    wandb_run.log(row, step=epoch)
                if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                    print(
                        f"epoch={epoch:04d} train_loss={row['train_loss']:.8f} "
                        f"val_loss={row['val_loss']:.8f} linear={row['val_linear_loss']:.8f} "
                        f"pred={row['val_pred_loss']:.8f} latent_std_mean={row['val_latent_std_mean']:.4f} "
                        f"spectral_radius={row['spectral_radius']:.4f}"
                    )
                if not np.isfinite(row["train_loss"]) or not np.isfinite(row["val_loss"]):
                    raise FloatingPointError(f"NaN/Inf detected at epoch {epoch}: {row}")
                if row["val_loss"] < best_val_loss - args.min_delta:
                    best_val_loss = row["val_loss"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, config, metadata)
                    if wandb_run is not None:
                        wandb_run.summary["best_epoch"] = best_epoch
                        wandb_run.summary["best_val_loss"] = best_val_loss
                else:
                    epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch} best_val_loss={best_val_loss:.8f}")
                    break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    save_checkpoint(last_path, model, optimizer, completed_epoch, best_val_loss, config, metadata)
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")
    print(f"best_epoch={best_epoch} best_val_loss={best_val_loss:.8f}")
    return best_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 10 Hz Full-A history-context Koopman v2.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--state-indices", type=str, default="0:12")
    parser.add_argument("--pressure-indices", type=str, default="0:12")
    parser.add_argument("--source-hz", type=float, default=10.0)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--upsample-factor", type=int, default=1)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--ksteps", type=int, default=50)
    parser.add_argument("--encode-dim", type=int, default=12)
    parser.add_argument("--hidden-sizes", type=str, default="128,128,64")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-std", type=float, default=1.0)
    parser.add_argument("--koopman-lam", type=float, default=10.0)
    parser.add_argument("--pred-lam", type=float, default=1.0)
    parser.add_argument("--stability-lam", type=float, default=0.01)
    parser.add_argument("--std-lam", type=float, default=0.1)
    parser.add_argument("--identity-lam", type=float, default=1e-4)
    parser.add_argument("--svd-lam", type=float, default=0.0)
    parser.add_argument("--augment-lam", type=float, default=0.0)
    parser.add_argument("--svd-min-singular-value", type=float, default=0.0)
    parser.add_argument("--spectral-radius-limit", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="soft-robot-koopman")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="fullA,history,10hz,koopman,v2")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    train(args)


if __name__ == "__main__":
    main()
