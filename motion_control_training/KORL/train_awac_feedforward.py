from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from .data import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_OUTPUT_ROOT,
        build_transition_dataset,
        dataset_action_bounds,
        load_episode_arrays,
        load_lerobot_state_stats,
        make_output_dir,
        normalize_rewards,
        parse_float_list,
        parse_hidden_sizes,
        parse_int_list,
        resolve_device,
        save_json,
        set_seed,
        split_episodes,
    )
except ImportError:  # pragma: no cover - direct script execution
    from data import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_OUTPUT_ROOT,
        build_transition_dataset,
        dataset_action_bounds,
        load_episode_arrays,
        load_lerobot_state_stats,
        make_output_dir,
        normalize_rewards,
        parse_float_list,
        parse_hidden_sizes,
        parse_int_list,
        resolve_device,
        save_json,
        set_seed,
        split_episodes,
    )


class ReplayBuffer:
    def __init__(self, dataset: dict[str, np.ndarray], device: torch.device) -> None:
        self.size = int(dataset["observations"].shape[0])
        self.states = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=device)
        self.rewards = torch.as_tensor(dataset["rewards"], dtype=torch.float32, device=device)
        self.next_states = torch.as_tensor(dataset["next_observations"], dtype=torch.float32, device=device)
        self.dones = torch.as_tensor(dataset["terminals"], dtype=torch.float32, device=device)

    def sample(self, batch_size: int) -> list[torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,), device=self.states.device)
        return [
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        ]


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        target_dim = state_dim // 2
        sizes = [target_dim] + hidden_sizes + [action_dim]
        layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.state_dim = int(state_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def target_state(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., self.state_dim // 2 :]

    def policy(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(self.target_state(state))
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        return torch.distributions.Normal(mean, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.policy(state).log_prob(action).sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor, deterministic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.policy(state)
        action = dist.mean if deterministic else dist.rsample()
        action = torch.max(torch.min(action, self.action_high.to(action.device)), self.action_low.to(action.device))
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        sizes = [state_dim + action_dim] + hidden_sizes + [1]
        layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AWAC:
    def __init__(
        self,
        actor: Actor,
        critic_1: Critic,
        critic_2: Critic,
        actor_optimizer: torch.optim.Optimizer,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float,
        tau: float,
        awac_lambda: float,
        exp_adv_max: float,
    ) -> None:
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = deepcopy(critic_1)
        self.target_critic_2 = deepcopy(critic_2)
        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.awac_lambda = float(awac_lambda)
        self.exp_adv_max = float(exp_adv_max)

    def actor_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pi_action, _ = self.actor(states)
            v = torch.min(self.critic_1(states, pi_action), self.critic_2(states, pi_action))
            q = torch.min(self.critic_1(states, actions), self.critic_2(states, actions))
            weights = torch.clamp_max(torch.exp((q - v) / self.awac_lambda), self.exp_adv_max)
        return (-self.actor.log_prob(states, actions) * weights).mean()

    def critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            q_next = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self.gamma * (1.0 - dones) * q_next
        q1_loss = nn.functional.mse_loss(self.critic_1(states, actions), q_target)
        q2_loss = nn.functional.mse_loss(self.critic_2(states, actions), q_target)
        return q1_loss + q2_loss

    def update(self, batch: list[torch.Tensor]) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        actor_loss = self.actor_loss(states, actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
        }


@torch.no_grad()
def evaluate(actor: Actor, dataset: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, Any]:
    actor.eval()
    actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=device)
    states = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=device)
    sq_error_sum: torch.Tensor | None = None
    abs_error_sum: torch.Tensor | None = None
    count = 0
    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        pred, _ = actor(states[start:end], deterministic=True)
        err = pred - actions[start:end]
        sq_error = err.pow(2).sum(dim=0)
        abs_error = err.abs().sum(dim=0)
        sq_error_sum = sq_error if sq_error_sum is None else sq_error_sum + sq_error
        abs_error_sum = abs_error if abs_error_sum is None else abs_error_sum + abs_error
        count += end - start
    rmse = torch.sqrt(sq_error_sum / max(count, 1)).detach().cpu().numpy()
    mae = (abs_error_sum / max(count, 1)).detach().cpu().numpy()
    actor.train()
    return {
        "action_rmse_mean": float(rmse.mean()),
        "action_mae_mean": float(mae.mean()),
        "action_rmse": rmse.tolist(),
        "action_mae": mae.tolist(),
    }


def save_checkpoint(
    path: Path,
    trainer: AWAC,
    step: int,
    config: dict[str, Any],
    metadata: dict[str, Any],
    eval_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "trainer_state_dict": trainer.state_dict(),
            "step": int(step),
            "config": config,
            "metadata": metadata,
            "eval": eval_metrics,
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    device = torch.device(resolve_device(args.device))
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = args.dataset_root.resolve()
    state_mean_full, state_std_full, state_names = load_lerobot_state_stats(dataset_root)
    state_indices = parse_int_list(args.state_indices, total_dim=len(state_mean_full))
    pressure_indices = parse_int_list(args.pressure_indices)
    state_mean = state_mean_full[state_indices]
    state_std = np.maximum(state_std_full[state_indices], args.norm_eps)
    episodes, pressure_meta = load_episode_arrays(dataset_root, state_indices, pressure_indices)
    train_episodes, val_episodes = split_episodes(sorted(episodes), args.val_ratio, args.seed)
    train_dataset, train_stats = build_transition_dataset(
        episodes,
        train_episodes,
        state_mean,
        state_std,
        args.target_offset,
        args.reward_scale,
        parse_float_list(args.reward_state_weights),
    )
    val_dataset, val_stats = build_transition_dataset(
        episodes,
        val_episodes,
        state_mean,
        state_std,
        args.target_offset,
        args.reward_scale,
        parse_float_list(args.reward_state_weights),
    )
    reward_norm_stats = normalize_rewards(train_dataset) if args.normalize_reward else {}
    if args.normalize_reward and reward_norm_stats["reward_std_before_norm"] > 1e-6:
        val_dataset["rewards"] = (
            val_dataset["rewards"] - reward_norm_stats["reward_mean_before_norm"]
        ) / reward_norm_stats["reward_std_before_norm"]

    action_low, action_high = dataset_action_bounds(train_dataset)
    action_low = np.minimum(action_low, 0.0).astype(np.float32)
    action_high = np.maximum(action_high, 1.0).astype(np.float32)
    output_dir = make_output_dir(args.output_root.resolve(), args.run_name)

    config = vars(args).copy()
    config["dataset_root"] = str(dataset_root)
    config["output_root"] = str(args.output_root.resolve())
    metadata = {
        "algorithm": "AWAC feedforward policy adapted from reference/KORL/training/awac_Feedforward.py",
        "observation_layout": "[normalized_current_state, normalized_future_target_state]",
        "target_offset": int(args.target_offset),
        "state_indices": state_indices,
        "state_names": [state_names[i] for i in state_indices],
        "pressure_indices": pressure_indices,
        "pressure_columns": pressure_meta["pressure_columns"],
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_dataset": train_stats,
        "val_dataset": val_stats,
        "reward_normalization": reward_norm_stats,
        "reward_type": train_stats["reward_type"],
        "reward_state_weights": train_stats["reward_state_weights"],
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
    }
    save_json(output_dir / "config.json", {"config": config, "metadata": metadata})

    state_dim = train_dataset["observations"].shape[1]
    action_dim = train_dataset["actions"].shape[1]
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    actor = Actor(state_dim, action_dim, hidden_sizes, action_low, action_high).to(device)
    critic_1 = Critic(state_dim, action_dim, hidden_sizes).to(device)
    critic_2 = Critic(state_dim, action_dim, hidden_sizes).to(device)
    trainer = AWAC(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=args.learning_rate),
        critic_1_optimizer=torch.optim.Adam(critic_1.parameters(), lr=args.learning_rate),
        critic_2_optimizer=torch.optim.Adam(critic_2.parameters(), lr=args.learning_rate),
        gamma=args.gamma,
        tau=args.tau,
        awac_lambda=args.awac_lambda,
        exp_adv_max=args.exp_adv_max,
    )
    replay_buffer = ReplayBuffer(train_dataset, device)

    metrics_path = output_dir / "metrics.csv"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_eval = float("inf")
    last_eval: dict[str, Any] = {}
    fieldnames = ["step", "critic_loss", "actor_loss", "val_action_rmse_mean", "val_action_mae_mean", "step_seconds"]

    print(f"dataset_root={dataset_root}")
    print(f"device={device} train_transitions={train_stats['transitions']} val_transitions={val_stats['transitions']}")
    print(f"state_dim={state_dim} action_dim={action_dim} target_offset={args.target_offset}")

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(1, args.max_timesteps + 1):
            start = time.perf_counter()
            logs = trainer.update(replay_buffer.sample(args.batch_size))
            if step == 1 or step % args.eval_freq == 0 or step == args.max_timesteps:
                last_eval = evaluate(actor, val_dataset, device, args.eval_batch_size)
                row = {
                    "step": step,
                    "critic_loss": logs["critic_loss"],
                    "actor_loss": logs["actor_loss"],
                    "val_action_rmse_mean": last_eval["action_rmse_mean"],
                    "val_action_mae_mean": last_eval["action_mae_mean"],
                    "step_seconds": time.perf_counter() - start,
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"step={step:06d} critic_loss={logs['critic_loss']:.6f} "
                    f"actor_loss={logs['actor_loss']:.6f} val_rmse={last_eval['action_rmse_mean']:.6f}"
                )
                if last_eval["action_rmse_mean"] < best_eval:
                    best_eval = last_eval["action_rmse_mean"]
                    save_checkpoint(best_path, trainer, step, config, metadata, last_eval)

    save_checkpoint(last_path, trainer, args.max_timesteps, config, metadata, last_eval)
    save_json(output_dir / "eval.json", {"best_val_action_rmse_mean": best_eval, "last_eval": last_eval})
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")
    return best_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train KORL AWAC feedforward policy from LeRobot data.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "feedforward")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--state-indices", type=str, default="0:12")
    parser.add_argument("--pressure-indices", type=str, default="0:12")
    parser.add_argument("--target-offset", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-state-weights", type=str, default="1,1,1,1,1,1,0,0,0,0,0,0")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--hidden-sizes", type=str, default="256,256,256")
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--awac-lambda", type=float, default=0.1)
    parser.add_argument("--exp-adv-max", type=float, default=100.0)
    parser.add_argument("--normalize-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-timesteps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-freq", type=int, default=1000)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
