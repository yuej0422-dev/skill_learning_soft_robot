from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
    from .train_awac_feedforward import ReplayBuffer
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
    from train_awac_feedforward import ReplayBuffer


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


class Scalar(nn.Module):
    def __init__(self, init_value: float) -> None:
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class KoopmanEncoder(nn.Module):
    def __init__(self, encode_layers: list[int], u_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(encode_layers[:-1], encode_layers[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(encode_layers) - 2:
                layers.append(nn.ReLU())
        self.encode_net = nn.Sequential(*layers)
        self.encode_layers = [int(v) for v in encode_layers]
        self.n_koopman = int(encode_layers[0] + encode_layers[-1])
        self.u_dim = int(u_dim)
        self.num_real = int(np.mod(self.n_koopman, 2))
        self.num_complex_pair = int(self.n_koopman / 2)
        self.B = nn.Parameter(torch.randn(self.u_dim, self.n_koopman) * 0.1)
        self.A = nn.Parameter(torch.normal(0.0, 0.01, size=(self.n_koopman,)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def form_A_from_eigenvalues(self) -> torch.Tensor:
        temp_A = self.A.new_zeros((self.n_koopman, self.n_koopman))
        for i in range(self.num_complex_pair):
            idx = 2 * i
            real = self.A[idx]
            imaginary = self.A[idx + 1]
            block = self.A.new_zeros((2, 2))
            block[0, 0] = real
            block[0, 1] = imaginary
            block[1, 0] = -imaginary
            block[1, 1] = real
            temp_A[idx : idx + 2, idx : idx + 2] = block
        for i in range(self.num_real):
            idx = 2 * self.num_complex_pair + i
            temp_A[idx, idx] = self.A[idx]
        return temp_A


class GaussianPolicy(nn.Module):
    def __init__(self, no_sig: bool = True, log_std_min: float = -20.0, log_std_max: float = 2.0) -> None:
        super().__init__()
        self.no_sig = bool(no_sig)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = Normal(mean, log_std.exp())
        return torch.sum(dist.log_prob(sample), dim=-1)

    def forward(self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = Normal(mean, log_std.exp())
        action = mean if deterministic else dist.rsample()
        return action, torch.sum(dist.log_prob(action), dim=-1)


class KoopmanFeedforwardPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        koopman_encoder: KoopmanEncoder,
        action_low: np.ndarray,
        action_high: np.ndarray,
        log_std_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.koopman_encoder = koopman_encoder
        self.tar_phi_net = nn.Sequential(
            nn.Linear(koopman_encoder.n_koopman, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(0.0)
        self.gaussian = GaussianPolicy(no_sig=True)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def mean_log_std(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_tar = observations[..., self.state_dim // 2 :]
        phi_x_tar = self.koopman_encoder.encode(x_tar)
        mean = self.tar_phi_net(phi_x_tar)
        log_std = self.log_std_multiplier() * self.log_std_net(observations) + self.log_std_offset()
        return mean, log_std

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        mean, log_std = self.mean_log_std(observations)
        return self.gaussian.log_prob(mean, log_std, actions)

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        mean, log_std = self.mean_log_std(observations)
        actions, log_probs = self.gaussian(mean, log_std, deterministic)
        actions = torch.max(torch.min(actions, self.action_high.to(actions.device)), self.action_low.to(actions.device))
        return actions, log_probs


class FullyConnectedQFunction(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, n_hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(observation_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = actions.ndim == 3 and observations.ndim == 2
        batch_size = observations.shape[0]
        if multiple_actions:
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = torch.squeeze(self.network(torch.cat([observations, actions], dim=-1)), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class KoopmanInformedQFunction(nn.Module):
    def __init__(
        self,
        koopman_encoder: KoopmanEncoder,
        feedforward_policy: KoopmanFeedforwardPolicy,
        state_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.koopman_encoder = koopman_encoder
        self.feedforward_policy = feedforward_policy
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_koopman = koopman_encoder.n_koopman
        self.z_dim = int((self.n_koopman + action_dim) * (self.n_koopman + action_dim + 1) / 2)
        self.h = nn.Parameter(torch.zeros(self.z_dim, 1))
        self._cached_h: torch.Tensor | None = None
        self._cached_k: torch.Tensor | None = None

    @property
    def h_cost(self) -> torch.Tensor:
        return -self.h

    def compute_h_and_k(self) -> tuple[torch.Tensor, torch.Tensor]:
        dim = self.n_koopman + self.action_dim
        h_mat = self.h.new_zeros((dim, dim))
        iu = torch.triu_indices(dim, dim, device=self.h.device)
        h_vec = self.h_cost.squeeze(-1).clone()
        diag_mask = iu[0] == iu[1]
        h_vec[~diag_mask] = h_vec[~diag_mask] / 2
        h_mat[iu[0], iu[1]] = h_vec
        h_mat = h_mat + h_mat.T - torch.diag(h_mat.diagonal())
        h_ae = h_mat[self.n_koopman :, : self.n_koopman]
        h_aa = h_mat[self.n_koopman :, self.n_koopman :]
        try:
            k = torch.linalg.solve(h_aa, h_ae)
        except RuntimeError:
            k = torch.linalg.pinv(h_aa) @ h_ae
        return h_mat, k.detach()

    def get_h_and_k(self) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            self._cached_h, self._cached_k = self.compute_h_and_k()
            return self._cached_h, self._cached_k
        if self._cached_h is None or self._cached_k is None:
            self._cached_h, self._cached_k = self.compute_h_and_k()
        return self._cached_h, self._cached_k

    def reset_cache(self) -> None:
        self._cached_h = None
        self._cached_k = None

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = actions.ndim == 3 and observations.ndim == 2
        batch_size = observations.shape[0]
        if multiple_actions:
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        x_cur = observations[:, : self.state_dim // 2]
        x_tar = observations[:, self.state_dim // 2 :]
        with torch.no_grad():
            phi_cur = self.koopman_encoder.encode(x_cur)
            phi_tar = self.koopman_encoder.encode(x_tar)
        with torch.no_grad():
            a_ff, _ = self.feedforward_policy(observations, deterministic=True)
        z = torch.cat([phi_cur - phi_tar, actions - a_ff], dim=-1)
        h_mat, _ = self.get_h_and_k()
        q_values = torch.einsum("bi,ij,bj->b", z, h_mat, z).unsqueeze(-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    def get_feedback_action(self, observations: torch.Tensor, repeat: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h_mat, k = self.get_h_and_k()
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            batch, n_repeat, _ = observations.shape
            x_cur = observations[:, :, : self.state_dim // 2]
            x_tar = observations[:, :, self.state_dim // 2 :]
            with torch.no_grad():
                phi_cur = self.koopman_encoder.encode(x_cur)
                phi_tar = self.koopman_encoder.encode(x_tar)
            error = phi_cur - phi_tar
            a_fb = -torch.einsum("bij,jk->bik", error, k.T).detach()
            z = torch.cat([error, a_fb], dim=-1).reshape(-1, error.shape[-1] + self.action_dim)
            q_values = torch.einsum("bi,ij,bj->b", z, h_mat, z).view(batch, n_repeat)
            return a_fb, q_values
        x_cur = observations[:, : self.state_dim // 2]
        x_tar = observations[:, self.state_dim // 2 :]
        with torch.no_grad():
            phi_cur = self.koopman_encoder.encode(x_cur)
            phi_tar = self.koopman_encoder.encode(x_tar)
        error = phi_cur - phi_tar
        a_fb = -torch.matmul(error, k.T).detach()
        z = torch.cat([error, a_fb], dim=-1)
        q_values = torch.einsum("bi,ij,bj->b", z, h_mat, z).unsqueeze(-1)
        return a_fb, q_values


class KORLTrainer:
    def __init__(
        self,
        actor: KoopmanFeedforwardPolicy,
        critic_1: FullyConnectedQFunction,
        critic_2: FullyConnectedQFunction,
        critic_1_kop: KoopmanInformedQFunction,
        koopman_encoder: KoopmanEncoder,
        optimizers: dict[str, torch.optim.Optimizer],
        q_mat: torch.Tensor,
        r_mat: torch.Tensor,
        args: argparse.Namespace,
    ) -> None:
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = deepcopy(critic_1).to(args.device_resolved)
        self.target_critic_2 = deepcopy(critic_2).to(args.device_resolved)
        self.critic_1_kop = critic_1_kop
        self.target_critic_1_kop = deepcopy(critic_1_kop).to(args.device_resolved)
        self.koopman_encoder = koopman_encoder
        self.optimizers = optimizers
        self.q_mat = q_mat
        self.r_mat = r_mat
        self.discount = args.discount
        self.soft_target_update_rate = args.soft_target_update_rate
        self.target_update_period = args.target_update_period
        self.cql_n_actions = args.cql_n_actions
        self.cql_temp = args.cql_temp
        self.cql_alpha = args.cql_alpha
        self.cql_target_action_gap = args.cql_target_action_gap
        self.cql_lagrange = args.cql_lagrange
        self.cql_clip_diff_min = args.cql_clip_diff_min
        self.cql_clip_diff_max = args.cql_clip_diff_max
        self.alpha_multiplier = args.alpha_multiplier
        self.backup_entropy = args.backup_entropy
        self.bc_steps = args.bc_steps
        self.actor_update_num = args.actor_update_num
        self.encode_update_num = args.encode_update_num
        self.total_it = 0
        self.log_alpha = Scalar(0.0).to(args.device_resolved)
        self.log_alpha_prime = Scalar(1.0).to(args.device_resolved)
        self.log_alpha_kop_prime = Scalar(1.0).to(args.device_resolved)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=args.policy_lr)
        self.alpha_prime_optimizer = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=args.qf_lr)
        self.alpha_kop_prime_optimizer = torch.optim.Adam(self.log_alpha_kop_prime.parameters(), lr=args.qf_lr * 2)
        self.target_entropy = -float(args.action_dim)
        self.update_phase = "encode"
        self.phase_step = 0

    def update_target_network(self) -> None:
        soft_update(self.target_critic_1, self.critic_1, self.soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, self.soft_target_update_rate)
        soft_update(self.target_critic_1_kop, self.critic_1_kop, self.soft_target_update_rate)
        self.target_critic_1_kop.reset_cache()

    def update_koopman_encoder(self, observations: torch.Tensor, actions: torch.Tensor, next_observations: torch.Tensor) -> torch.Tensor:
        n_state = observations.shape[1] // 2
        x_cur = observations[:, :n_state]
        x_next = next_observations[:, :n_state]
        phi_x_cur = self.koopman_encoder.encode(x_cur)
        phi_x_next = self.koopman_encoder.encode(x_next)
        pred_phi_x_next = phi_x_cur @ self.koopman_encoder.form_A_from_eigenvalues() + actions @ self.koopman_encoder.B
        loss = F.mse_loss(pred_phi_x_next, phi_x_next)
        self.optimizers["koopman"].zero_grad(set_to_none=True)
        loss.backward()
        self.optimizers["koopman"].step()
        return loss.detach()

    def alpha_and_alpha_loss(self, log_pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
        alpha = self.log_alpha().exp() * self.alpha_multiplier
        return alpha, alpha_loss

    def q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        logs: dict[str, float],
    ) -> torch.Tensor:
        q1 = self.critic_1(observations, actions)
        q2 = self.critic_2(observations, actions)
        new_next_actions, next_log_pi = self.actor(next_observations)
        target_q = torch.min(self.target_critic_1(next_observations, new_next_actions), self.target_critic_2(next_observations, new_next_actions))
        if self.backup_entropy:
            target_q = target_q - alpha * next_log_pi
        td_target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * self.discount * target_q.detach()
        q_loss_base = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        batch_size, action_dim = actions.shape
        rand_actions = actions.new_empty((batch_size, self.cql_n_actions, action_dim)).uniform_(0, 1)
        cur_actions, cur_log_pi = self.actor(observations, repeat=self.cql_n_actions)
        next_actions, next_log_pi = self.actor(next_observations, repeat=self.cql_n_actions)
        random_density = np.log(0.5**action_dim)
        q1_cat = torch.cat(
            [
                self.critic_1(observations, rand_actions) - random_density,
                self.critic_1(observations, next_actions.detach()) - next_log_pi.detach(),
                self.critic_1(observations, cur_actions.detach()) - cur_log_pi.detach(),
            ],
            dim=1,
        )
        q2_cat = torch.cat(
            [
                self.critic_2(observations, rand_actions) - random_density,
                self.critic_2(observations, next_actions.detach()) - next_log_pi.detach(),
                self.critic_2(observations, cur_actions.detach()) - cur_log_pi.detach(),
            ],
            dim=1,
        )
        cql_q1_ood = torch.logsumexp(q1_cat / self.cql_temp, dim=1) * self.cql_temp
        cql_q2_ood = torch.logsumexp(q2_cat / self.cql_temp, dim=1) * self.cql_temp
        cql_q1_diff = torch.clamp(cql_q1_ood - q1, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        cql_q2_diff = torch.clamp(cql_q2_ood - q2, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        cql_loss = self.cql_alpha * (cql_q1_diff + cql_q2_diff)
        logs.update(qf_loss=float(q_loss_base.item()), cql_qf1_diff=float(cql_q1_diff.item()), cql_qf2_diff=float(cql_q2_diff.item()))
        return q_loss_base + cql_loss

    def feedback_rewards(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n_state = observations.shape[1] // 2
            x_cur = observations[:, :n_state]
            x_tar = observations[:, n_state:]
            phi_cur = self.koopman_encoder.encode(x_cur)
            phi_tar = self.koopman_encoder.encode(x_tar)
            a_ff, _ = self.actor(observations, deterministic=True)
            a_fb = actions - a_ff
            cost_state = torch.einsum("bi,ij,bj->b", phi_cur - phi_tar, self.q_mat, phi_cur - phi_tar)
            cost_action = torch.einsum("bi,ij,bj->b", a_fb, self.r_mat, a_fb)
            return -(cost_state + cost_action).reshape(-1, 1)

    def koopman_q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        logs: dict[str, float],
    ) -> torch.Tensor:
        q1_kop = -self.critic_1_kop(observations, actions)
        with torch.no_grad():
            _, target_q = self.target_critic_1_kop.get_feedback_action(next_observations)
            target_q = -target_q
        td_target = rewards + (1.0 - dones) * self.discount * target_q.detach()
        q_loss = F.mse_loss(q1_kop, td_target)
        rand_actions = actions.new_empty((actions.shape[0], self.cql_n_actions, actions.shape[1])).uniform_(0, 1)
        q1_rand = -self.critic_1_kop(observations, rand_actions)
        _, q1_cur = self.critic_1_kop.get_feedback_action(observations, repeat=self.cql_n_actions)
        _, q1_next = self.critic_1_kop.get_feedback_action(next_observations, repeat=self.cql_n_actions)
        q_cat = torch.cat([q1_rand, q1_kop, -q1_cur, -q1_next], dim=1)
        cql_q_ood = torch.logsumexp(q_cat / self.cql_temp, dim=1) * self.cql_temp
        cql_diff = torch.clamp(cql_q_ood - q1_kop, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        cql_loss = self.cql_alpha * cql_diff
        logs.update(kop_qf1_loss=float(q_loss.item()), kop_cql_qf1_diff=float(cql_diff.item()))
        return q_loss + cql_loss

    def train(self, batch: list[torch.Tensor]) -> dict[str, float]:
        observations, actions, rewards, next_observations, dones = batch
        self.total_it += 1
        logs: dict[str, float] = {}
        if self.update_phase == "encode":
            loss = self.update_koopman_encoder(observations, actions, next_observations)
            logs["async_linear_loss"] = float(loss.item())
            self.phase_step += 1
            if self.phase_step >= self.encode_update_num:
                self.update_phase = "actor"
                self.phase_step = 0
            return logs

        new_actions, log_pi = self.actor(observations)
        alpha, alpha_loss = self.alpha_and_alpha_loss(log_pi)
        if self.total_it <= self.bc_steps:
            policy_loss = (alpha * log_pi - self.actor.log_prob(observations, actions)).mean()
        else:
            q_new_actions = torch.min(self.critic_1(observations, new_actions), self.critic_2(observations, new_actions))
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        q_loss = self.q_loss(observations, actions, next_observations, rewards, dones, alpha, logs)
        fb_rewards = self.feedback_rewards(observations, actions)
        kop_loss = self.koopman_q_loss(observations, actions, next_observations, fb_rewards, dones, logs)

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.optimizers["actor"].zero_grad(set_to_none=True)
        policy_loss.backward(retain_graph=True)
        self.optimizers["actor"].step()
        self.optimizers["critic_1"].zero_grad(set_to_none=True)
        self.optimizers["critic_2"].zero_grad(set_to_none=True)
        q_loss.backward(retain_graph=True)
        self.optimizers["critic_1"].step()
        self.optimizers["critic_2"].step()
        self.optimizers["critic_1_kop"].zero_grad(set_to_none=True)
        kop_loss.backward()
        self.optimizers["critic_1_kop"].step()
        self.critic_1_kop.reset_cache()
        if self.total_it % self.target_update_period == 0:
            self.update_target_network()
        self.phase_step += 1
        if self.phase_step >= self.actor_update_num:
            self.update_phase = "encode"
            self.phase_step = 0
        logs.update(policy_loss=float(policy_loss.item()), alpha=float(alpha.item()))
        return logs

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_kop": self.critic_1_kop.state_dict(),
            "koopman_encoder": self.koopman_encoder.state_dict(),
            "total_it": self.total_it,
        }


@torch.no_grad()
def eval_encoder_linear(koopman_encoder: KoopmanEncoder, dataset: dict[str, np.ndarray], device: torch.device) -> float:
    observations = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=device)
    next_observations = torch.as_tensor(dataset["next_observations"], dtype=torch.float32, device=device)
    n_state = observations.shape[1] // 2
    losses = []
    for start in range(0, observations.shape[0], 4096):
        end = min(start + 4096, observations.shape[0])
        phi_cur = koopman_encoder.encode(observations[start:end, :n_state])
        phi_next = koopman_encoder.encode(next_observations[start:end, :n_state])
        pred = phi_cur @ koopman_encoder.form_A_from_eigenvalues() + actions[start:end] @ koopman_encoder.B
        losses.append(F.mse_loss(pred, phi_next, reduction="sum").item())
    return float(sum(losses) / max(observations.shape[0], 1) / koopman_encoder.n_koopman)


@torch.no_grad()
def evaluate_policy(
    trainer: KORLTrainer,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    trainer.actor.eval()
    observations = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.float32, device=device)
    ff_sq = None
    total_sq = None
    fb_abs = None
    count = 0
    low = trainer.actor.action_low.to(device)
    high = trainer.actor.action_high.to(device)
    for start in range(0, observations.shape[0], batch_size):
        end = min(start + batch_size, observations.shape[0])
        obs = observations[start:end]
        ff, _ = trainer.actor(obs, deterministic=True)
        fb, _ = trainer.critic_1_kop.get_feedback_action(obs)
        total = torch.max(torch.min(ff + fb, high), low)
        ff_err = (ff - actions[start:end]).pow(2).sum(dim=0)
        total_err = (total - actions[start:end]).pow(2).sum(dim=0)
        fb_val = fb.abs().sum(dim=0)
        ff_sq = ff_err if ff_sq is None else ff_sq + ff_err
        total_sq = total_err if total_sq is None else total_sq + total_err
        fb_abs = fb_val if fb_abs is None else fb_abs + fb_val
        count += end - start
    ff_rmse = torch.sqrt(ff_sq / max(count, 1)).cpu().numpy()
    total_rmse = torch.sqrt(total_sq / max(count, 1)).cpu().numpy()
    mean_fb_abs = (fb_abs / max(count, 1)).cpu().numpy()
    trainer.actor.train()
    return {
        "ff_action_rmse_mean": float(ff_rmse.mean()),
        "feedback_total_action_rmse_mean": float(total_rmse.mean()),
        "feedback_abs_mean": float(mean_fb_abs.mean()),
        "ff_action_rmse": ff_rmse.tolist(),
        "feedback_total_action_rmse": total_rmse.tolist(),
        "feedback_abs": mean_fb_abs.tolist(),
    }


def load_feedforward_actor_weights(path: Path, actor: KoopmanFeedforwardPolicy) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint["trainer_state_dict"]["actor"]
    target = actor.state_dict()
    mapped = {}
    for key, value in source.items():
        if key == "net.0.weight":
            target_key = "tar_phi_net.0.weight"
            if target_key in target and target[target_key].shape[0] == value.shape[0]:
                expanded = target[target_key].clone()
                expanded.zero_()
                expanded[:, : value.shape[1]] = value
                mapped[target_key] = expanded
        elif key.startswith("net."):
            mapped["tar_phi_net." + key[len("net.") :]] = value
    compatible = {key: value for key, value in mapped.items() if key in target and target[key].shape == value.shape}
    target.update(compatible)
    actor.load_state_dict(target)
    print(f"Loaded {len(compatible)} compatible feedforward tensors from {path}")


def train(args: argparse.Namespace) -> Path:
    args.device_resolved = resolve_device(args.device)
    device = torch.device(args.device_resolved)
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
    reward_state_weights = parse_float_list(args.reward_state_weights)
    train_dataset, train_stats = build_transition_dataset(
        episodes,
        train_episodes,
        state_mean,
        state_std,
        args.target_offset,
        args.reward_scale,
        reward_state_weights,
    )
    val_dataset, val_stats = build_transition_dataset(
        episodes,
        val_episodes,
        state_mean,
        state_std,
        args.target_offset,
        args.reward_scale,
        reward_state_weights,
    )
    reward_norm_stats = normalize_rewards(train_dataset) if args.normalize_reward else {}
    if args.normalize_reward and reward_norm_stats["reward_std_before_norm"] > 1e-6:
        val_dataset["rewards"] = (val_dataset["rewards"] - reward_norm_stats["reward_mean_before_norm"]) / reward_norm_stats["reward_std_before_norm"]
    action_low, action_high = dataset_action_bounds(train_dataset)
    action_low = np.minimum(action_low, 0.0).astype(np.float32)
    action_high = np.maximum(action_high, 1.0).astype(np.float32)

    state_dim = train_dataset["observations"].shape[1]
    action_dim = train_dataset["actions"].shape[1]
    args.action_dim = action_dim
    hidden_sizes = parse_hidden_sizes(args.koopman_hidden_sizes)
    encode_layers = [state_dim // 2] + hidden_sizes + [args.koopman_encode_dim]
    koopman_encoder = KoopmanEncoder(encode_layers, action_dim).to(device)
    actor = KoopmanFeedforwardPolicy(state_dim, action_dim, args.hidden_dim, koopman_encoder, action_low, action_high).to(device)
    if args.feedforward_checkpoint:
        load_feedforward_actor_weights(args.feedforward_checkpoint, actor)
    critic_1 = FullyConnectedQFunction(state_dim, action_dim, args.hidden_dim, args.q_n_hidden_layers).to(device)
    critic_2 = FullyConnectedQFunction(state_dim, action_dim, args.hidden_dim, args.q_n_hidden_layers).to(device)
    critic_1_kop = KoopmanInformedQFunction(koopman_encoder, actor, state_dim, action_dim).to(device)
    optimizers = {
        "actor": torch.optim.Adam(actor.parameters(), lr=args.policy_lr),
        "critic_1": torch.optim.Adam(critic_1.parameters(), lr=args.qf_lr),
        "critic_2": torch.optim.Adam(critic_2.parameters(), lr=args.qf_lr),
        "critic_1_kop": torch.optim.Adam([critic_1_kop.h], lr=args.kop_qf_lr, weight_decay=1e-5),
        "koopman": torch.optim.Adam(koopman_encoder.parameters(), lr=args.k_lr),
    }
    q_diag = np.full(koopman_encoder.n_koopman, args.q_diag_value, dtype=np.float32)
    r_diag = np.full(action_dim, args.r_diag_value, dtype=np.float32)
    q_mat = torch.diag(torch.as_tensor(q_diag, dtype=torch.float32, device=device))
    r_mat = torch.diag(torch.as_tensor(r_diag, dtype=torch.float32, device=device))
    trainer = KORLTrainer(actor, critic_1, critic_2, critic_1_kop, koopman_encoder, optimizers, q_mat, r_mat, args)
    replay_buffer = ReplayBuffer(train_dataset, device)
    output_dir = make_output_dir(args.output_root.resolve(), args.run_name)

    config = vars(args).copy()
    config["dataset_root"] = str(dataset_root)
    config["output_root"] = str(args.output_root.resolve())
    metadata = {
        "algorithm": "KORL CQL feedback with synchronous Koopman updates adapted from reference/KORL/training/cql_kop_Async_Qlinear.py",
        "observation_layout": "[normalized_current_state, normalized_future_target_state]",
        "target_offset": int(args.target_offset),
        "koopman_encode_layers": encode_layers,
        "state_indices": state_indices,
        "state_names": [state_names[i] for i in state_indices],
        "pressure_indices": pressure_indices,
        "pressure_columns": pressure_meta["pressure_columns"],
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_dataset": train_stats,
        "val_dataset": val_stats,
        "reward_normalization": reward_norm_stats,
        "transition_reward_type": train_stats["reward_type"],
        "transition_reward_state_weights": train_stats["reward_state_weights"],
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
    }
    save_json(output_dir / "config.json", {"config": config, "metadata": metadata})
    print(f"dataset_root={dataset_root}")
    print(f"device={device} train_transitions={train_stats['transitions']} val_transitions={val_stats['transitions']}")
    print(f"state_dim={state_dim} action_dim={action_dim} encode_layers={encode_layers}")

    metrics_path = output_dir / "metrics.csv"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_eval = float("inf")
    last_eval: dict[str, Any] = {}
    fieldnames = [
        "step",
        "phase",
        "async_linear_loss",
        "policy_loss",
        "qf_loss",
        "kop_qf1_loss",
        "val_ff_action_rmse_mean",
        "val_feedback_total_action_rmse_mean",
        "val_koopman_linear_mse",
        "step_seconds",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(1, args.max_timesteps + 1):
            start = time.perf_counter()
            logs = trainer.train(replay_buffer.sample(args.batch_size))
            if step == 1 or step % args.eval_freq == 0 or step == args.max_timesteps:
                last_eval = evaluate_policy(trainer, val_dataset, device, args.eval_batch_size)
                linear_mse = eval_encoder_linear(koopman_encoder, val_dataset, device)
                row = {
                    "step": step,
                    "phase": trainer.update_phase,
                    "async_linear_loss": logs.get("async_linear_loss", ""),
                    "policy_loss": logs.get("policy_loss", ""),
                    "qf_loss": logs.get("qf_loss", ""),
                    "kop_qf1_loss": logs.get("kop_qf1_loss", ""),
                    "val_ff_action_rmse_mean": last_eval["ff_action_rmse_mean"],
                    "val_feedback_total_action_rmse_mean": last_eval["feedback_total_action_rmse_mean"],
                    "val_koopman_linear_mse": linear_mse,
                    "step_seconds": time.perf_counter() - start,
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"step={step:06d} phase={trainer.update_phase} "
                    f"ff_rmse={last_eval['ff_action_rmse_mean']:.6f} "
                    f"fb_total_rmse={last_eval['feedback_total_action_rmse_mean']:.6f} "
                    f"koopman_mse={linear_mse:.6f}"
                )
                score = last_eval["feedback_total_action_rmse_mean"]
                if score < best_eval:
                    best_eval = score
                    torch.save(
                        {
                            "trainer_state_dict": trainer.state_dict(),
                            "step": step,
                            "config": config,
                            "metadata": metadata,
                            "eval": {**last_eval, "koopman_linear_mse": linear_mse},
                        },
                        best_path,
                    )
    torch.save(
        {
            "trainer_state_dict": trainer.state_dict(),
            "step": args.max_timesteps,
            "config": config,
            "metadata": metadata,
            "eval": last_eval,
        },
        last_path,
    )
    save_json(output_dir / "eval.json", {"best_feedback_total_action_rmse_mean": best_eval, "last_eval": last_eval})
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")
    return best_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train KORL feedback policy from LeRobot data.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "feedback")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--feedforward-checkpoint", type=Path, default=None)
    parser.add_argument("--state-indices", type=str, default="0:12")
    parser.add_argument("--pressure-indices", type=str, default="0:12")
    parser.add_argument("--target-offset", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-state-weights", type=str, default="1,1,1,1,1,1,0,0,0,0,0,0")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--q-n-hidden-layers", type=int, default=3)
    parser.add_argument("--koopman-hidden-sizes", type=str, default="64,128,64")
    parser.add_argument("--koopman-encode-dim", type=int, default=12)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--qf-lr", type=float, default=1e-5)
    parser.add_argument("--kop-qf-lr", type=float, default=2e-5)
    parser.add_argument("--k-lr", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--soft-target-update-rate", type=float, default=0.005)
    parser.add_argument("--target-update-period", type=int, default=1)
    parser.add_argument("--alpha-multiplier", type=float, default=1.0)
    parser.add_argument("--backup-entropy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bc-steps", type=int, default=0)
    parser.add_argument("--cql-n-actions", type=int, default=10)
    parser.add_argument("--cql-temp", type=float, default=1.0)
    parser.add_argument("--cql-alpha", type=float, default=5.0)
    parser.add_argument("--cql-target-action-gap", type=float, default=0.8)
    parser.add_argument("--cql-lagrange", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cql-clip-diff-min", type=float, default=-200.0)
    parser.add_argument("--cql-clip-diff-max", type=float, default=float("inf"))
    parser.add_argument("--actor-update-num", type=int, default=10)
    parser.add_argument("--encode-update-num", type=int, default=10)
    parser.add_argument("--q-diag-value", type=float, default=0.5)
    parser.add_argument("--r-diag-value", type=float, default=10.0)
    parser.add_argument("--normalize-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-timesteps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-freq", type=int, default=1000)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
