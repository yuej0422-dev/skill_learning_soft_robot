# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# https://arxiv.org/pdf/2006.04779.pdf
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
# wandb.init(mode="offline")
from torch.distributions import Normal, TransformedDistribution, SigmoidTransform
from collections import OrderedDict
from dataclasses import field
import scipy.io
import math
import time
import copy
# import mujoco

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_alpha: float = 10.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    # AntMaze hacks
    bc_steps: int = int(0)  # Number of BC steps at start
    reward_scale: float = 5.0
    reward_bias: float = -1.0
    policy_log_std_multiplier: float = 1.0

    # Wandb logging
    project: str = "CORL"
    group: str = "CQL-D4RL"
    name: str = "CQL"

    # Addition
    dataset_path: str = ""
    horizon: int = 100
    Q_diag: List[float] = field(default_factory=list)
    R_diag: List[float] = field(default_factory=list)
    reg_coeff_fb: float = 0
    linear_coeff: float = 0
    k_lr: float = 1e-5
    actor_update_num: int = 1
    encode_update_num: int = 1
    koopman_encode_layers: List[int] = field(default_factory=lambda: [12, 64, 128, 64, 12])
    kop_qf_lr: float = 1e-5

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


# @torch.no_grad()
# def eval_actor(
#     env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
# ) -> np.ndarray:
#     env.seed(seed)
#     actor.eval()
#     episode_rewards = []
#     for _ in range(n_episodes):
#         state, done = env.reset(), False
#         episode_reward = 0.0
#         while not done:
#             action = actor.act(state, device)
#             state, reward, done, _ = env.step(action)
#             episode_reward += reward
#         episode_rewards.append(episode_reward)
#
#     actor.train()
#     return np.asarray(episode_rewards)


def quaternion_to_euler(quaternion, rotation_sequence='sxyz'):
    q0, q1, q2, q3 = quaternion
    # 计算旋转矩阵的元素
    R = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2- 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ])
    # 根据旋转顺序计算欧拉角
    if rotation_sequence == 'sxyz':
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        euler_x = math.atan2(R[2, 1], R[2, 2])
        euler_y = math.atan2(-R[2, 0], sy)
        euler_z = math.atan2(R[1, 0], R[0, 0])
    elif rotation_sequence == 'szyx':
        asiny = -2*q1*q3 + 2*q0*q2
        asiny = np.clip(asiny, -1, 1)
        euler_x = math.atan2(R[1, 0], q0**2 + q1**2 - q2**2 - q3**2)
        euler_y = math.asin(asiny)
        euler_z = math.atan2(R[2, 1], q0**2 - q1**2 - q2**2 + q3**2)
    else:
        raise ValueError("Unsupported rotation sequence.")

    return np.array([euler_x, euler_y, euler_z])


def mb_eval_reward_fn(x, target):
    # 计算 L2 距离（欧几里得距离）
    return -np.linalg.norm(x - target)


@torch.no_grad()
def eval_encoder_linear(
    Koopman_encoder: nn.Module,
    dataset: dict,
    device: str,
) -> float:
    """
    使用训练好的 Koopman 参数 (A, B) 评估 encode_net 的线性性：
    phi(x') ≈ phi(x)·A^T + u·B^T，返回 MSE 误差（越小越线性）。
    """

    observations = dataset["observations"]           # shape [N, 2 * obs_dim]
    actions = dataset["actions"]                     # shape [N, a_dim]
    next_observations = dataset["next_observations"]

    obs_dim = observations.shape[1] // 2
    x_cur = observations[:, :obs_dim]
    x_next = next_observations[:, :obs_dim]

    # 转为 tensor
    x_cur_tensor = torch.tensor(x_cur, dtype=torch.float32, device=device)
    x_next_tensor = torch.tensor(x_next, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

    # Koopman lifting
    phi_x_cur = Koopman_encoder.encode(x_cur_tensor)     # shape: [B, Nkoopman]
    phi_x_next = Koopman_encoder.encode(x_next_tensor)

    A_mat = Koopman_encoder.form_A_from_eigenvalues()    # shape: [N, N]
    B_mat = Koopman_encoder.B                            # shape: [a, N]

    # Koopman线性预测
    pred_phi_x_next = phi_x_cur @ A_mat + actions_tensor @ B_mat  # shape: [B, N]

    # MSE loss
    mse = nn.functional.mse_loss(pred_phi_x_next, phi_x_next).item()

    return mse


# def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
#     returns, lengths = [], []
#     ep_ret, ep_len = 0.0, 0
#     for r, d in zip(dataset["rewards"], dataset["terminals"]):
#         ep_ret += float(r)
#         ep_len += 1
#         if d or ep_len == max_episode_steps:
#             returns.append(ep_ret)
#             lengths.append(ep_len)
#             ep_ret, ep_len = 0.0, 0
#     lengths.append(ep_len)  # but still keep track of number of steps
#     assert sum(lengths) == len(dataset["rewards"])
#     return min(returns), max(returns)


# def modify_reward(
#     dataset: Dict,
#     env_name: str,
#     max_episode_steps: int = 1000,
#     reward_scale: float = 1.0,
#     reward_bias: float = 0.0,
# ):
#     if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
#         min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
#         dataset["rewards"] /= max_ret - min_ret
#         dataset["rewards"] *= max_episode_steps
#     dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
def modify_reward(dataset):
    reward_mean = dataset["rewards"].mean()
    reward_std = dataset["rewards"].std()

    dataset["rewards"] = (dataset["rewards"] - reward_mean) / reward_std
    return reward_mean, reward_std



def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


def vec_H(H: np.ndarray) -> np.ndarray:
    size = H.shape[0]
    newsize = int(size * (size + 1) / 2)
    vec = np.zeros((newsize,))
    n = 0
    for i in range(size):
        for j in range(i, size):
            if i == j:
                vec[n] = H[i, j]
            else:
                vec[n] = 2 * H[i, j]
            n += 1
    return vec.reshape(-1, 1)


def vec_z(z: np.ndarray) -> np.ndarray:
    size = z.shape[0]
    newsize = int(size * (size + 1) / 2)
    vec = np.zeros((newsize,))
    n = 0
    for i in range(size):
        for j in range(i, size):
            vec[n] = z[i] * z[j]
            n += 1
    return vec.reshape(-1, 1)


def vec_H_inv(h: np.ndarray) -> np.ndarray:
    n = h.shape[0]
    size = int((-1 + np.sqrt(1 + 8 * n)) // 2)
    H = np.zeros((size, size))
    idx = 0
    for i in range(size):
        for j in range(i, size):
            if i == j:
                H[i, j] = h[idx]
            else:
                H[i, j] = H[j, i] = h[idx] / 2
            idx += 1
    return H


class ReparameterizedSigmGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_sig: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_sig = no_sig

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_sig:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), SigmoidTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_sig:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), SigmoidTransform(cache_size=1)
            )

        if deterministic:
            action_sample = mean
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class KoopmanEncoder(nn.Module):
    def __init__(self, encode_layers, u_dim):
        super(KoopmanEncoder, self).__init__()

        self.Nkoopman = encode_layers[0] + encode_layers[-1]
        self.u_dim = u_dim

        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers[f"linear_{layer_i}"] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers[f"relu_{layer_i}"] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)

        self.num_real = int(np.mod(self.Nkoopman, 2))
        self.num_complex_pair = int(self.Nkoopman / 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize B
        wb = torch.randn(self.u_dim, self.Nkoopman) * 0.1
        self.B = nn.Parameter(wb, requires_grad=True)

        # Initialize A as eigenvalue vector
        init_A = torch.normal(0.0, 0.01, size=(self.Nkoopman,))
        self.A = nn.Parameter(init_A, requires_grad=True)

    def encode_only(self, x):
        return self.encode_net(x)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def forward(self, x, u):
        temp_A = self.form_A_from_eigenvalues().to(self.device)
        x_new = torch.matmul(x, temp_A)
        Bu = torch.matmul(u, self.B)
        return x_new + Bu

    def form_A_from_eigenvalues(self):
        assert self.Nkoopman == self.num_real + 2 * self.num_complex_pair, \
            "The sum of all eigenvalue blocks must equal Nkoopman"
        idx = 0
        temp_A = torch.zeros([self.Nkoopman, self.Nkoopman], device=self.A.device)
        for i in range(self.num_complex_pair):
            idx = 2 * i
            temp_A[idx:idx + 2, idx:idx + 2] = self.form_complex_conjugate_block(self.A[idx], self.A[idx + 1])
        for i in range(self.num_real):
            idx = 2 * self.num_complex_pair + i
            temp_A[idx, idx] = self.A[idx]
        return temp_A

    def form_complex_conjugate_block(self, real, imaginary):
        """
        构造复共轭特征值对应的 2x2 宏块（用于 Koopman A 的重构）
        注意：返回 torch.Tensor，确保设备兼容
        """
        block = torch.zeros((2, 2), dtype=torch.float32, device=real.device)
        block[0, 0] = real
        block[0, 1] = imaginary
        block[1, 0] = -imaginary
        block[1, 1] = real
        return block


# class KoopmanGaussianPolicy(nn.Module):
#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         max_action: float,
#         Koopman_encoder: nn.Module,
#         log_std_multiplier: float = 1.0,
#         log_std_offset: float = 0.0,
#         hidden_dim: int = 256,
#         no_sig: bool = True,
#     ):
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.max_action = max_action
#         self.no_sig = no_sig
#
#         self.Koopman_encoder = Koopman_encoder
#         self.Nkoopman = Koopman_encoder.Nkoopman
#
#         # 主体网络结构
#         self.cur_phi_layer = nn.Linear(self.Nkoopman, action_dim, bias=False)
#         self.tar_phi_net = nn.Sequential(
#             nn.Linear(self.Nkoopman, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#         )
#
#         self.log_std_net = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#         )
#
#         # 参数封装
#         self.log_std_multiplier = Scalar(log_std_multiplier)
#         self.log_std_offset = Scalar(log_std_offset)
#         self.sigmoid_gaussian = ReparameterizedSigmGaussian(no_sig=no_sig)
#
#     def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         if actions.ndim == 3:
#             observations = extend_and_repeat(observations, 1, actions.shape[1])
#
#         x_cur = observations[:, :self.state_dim // 2]
#         x_tar = observations[:, self.state_dim // 2:]
#         phi_x_cur = self.Koopman_encoder.encode(x_cur)
#         phi_x_tar = self.Koopman_encoder.encode(x_tar)
#
#         mean_cur = self.cur_phi_layer(phi_x_cur - phi_x_tar)
#         mean_tar = self.tar_phi_net(phi_x_tar)
#         mean = mean_cur + mean_tar
#
#         log_std = self.log_std_net(observations)
#         log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
#         _, log_probs = self.sigmoid_gaussian(mean, log_std, False)
#         return log_probs
#
#     def forward(self, observations: torch.Tensor, deterministic: bool = False, repeat: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         if repeat is not None:
#             observations = extend_and_repeat(observations, 1, repeat)
#
#         if repeat is None:
#             x_cur = observations[:, :self.state_dim // 2]
#             x_tar = observations[:, self.state_dim // 2:]
#         else:
#             x_cur = observations[:, :, :self.state_dim // 2]
#             x_tar = observations[:, :, self.state_dim // 2:]
#
#         phi_x_cur = self.Koopman_encoder.encode(x_cur)
#         phi_x_tar = self.Koopman_encoder.encode(x_tar)
#
#         mean_cur = self.cur_phi_layer(phi_x_cur - phi_x_tar)
#         mean_tar = self.tar_phi_net(phi_x_tar)
#         mean = mean_cur + mean_tar
#
#         log_std = self.log_std_net(observations)
#         log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
#         actions, log_probs = self.sigmoid_gaussian(mean, log_std, deterministic)
#         return actions, log_probs
#
#     @torch.no_grad()
#     def act(self, state: np.ndarray, device: str = "cpu"):
#         state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
#         with torch.no_grad():
#             actions, _ = self(state, not self.training)
#         return actions.cpu().data.numpy().flatten()


class KoopmanFeedforwardPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        Koopman_encoder: nn.Module,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = 0.0,
        hidden_dim: int = 256,
        no_sig: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.no_sig = no_sig

        self.Koopman_encoder = Koopman_encoder
        self.Nkoopman = Koopman_encoder.Nkoopman

        # 仅根据 x_tar 输出动作
        self.tar_phi_net = nn.Sequential(
            nn.Linear(self.Nkoopman, hidden_dim),
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
        self.log_std_offset = Scalar(log_std_offset)
        self.sigmoid_gaussian = ReparameterizedSigmGaussian(no_sig=no_sig)

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])

        x_tar = observations[:, self.state_dim // 2:]
        with torch.no_grad():
            phi_x_tar = self.Koopman_encoder.encode(x_tar)

        mean = self.tar_phi_net(phi_x_tar)

        log_std = self.log_std_net(observations)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.sigmoid_gaussian(mean, log_std, False)
        return log_probs

    def forward(self, observations: torch.Tensor, deterministic: bool = False, repeat: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)

        if repeat is None:
            x_tar = observations[:, self.state_dim // 2:]
        else:
            x_tar = observations[:, :, self.state_dim // 2:]

        with torch.no_grad():
            phi_x_tar = self.Koopman_encoder.encode(x_tar)
        mean = self.tar_phi_net(phi_x_tar)

        log_std = self.log_std_net(observations)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.sigmoid_gaussian(mean, log_std, deterministic)
        return actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class KoopmanInformedQFunction(nn.Module):
    def __init__(
        self,
        Koopman_encoder: nn.Module,
        Feedforward_policy: nn.Module,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.Koopman_encoder = Koopman_encoder
        self.Feedforward_policy = Feedforward_policy
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.Nkoopman = Koopman_encoder.Nkoopman
        self.z_dim = int((self.Nkoopman + action_dim) * (self.Nkoopman + action_dim + 1) / 2)

        # 初始化为向量化后的 H 矩阵（对称）
        self.h = nn.Parameter(torch.zeros(self.z_dim, 1))  # 向量化的 H

        self._cached_K = None  # 用于缓存 K
        self._cached_H = None  # 用于缓存 H

    @property
    def h_cost(self) -> torch.Tensor:
        """返回与控制一致的 H_cost 向量形式"""
        return -self.h

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]

        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])

        x_cur = observations[:, :self.state_dim // 2]
        x_tar = observations[:, self.state_dim // 2:]

        phi_cur = self.Koopman_encoder.encode(x_cur)
        phi_tar = self.Koopman_encoder.encode(x_tar)
        e = phi_cur - phi_tar  # 升维误差

        # 前馈控制动作（不需要梯度）
        with torch.no_grad():
            a_ff, _ = self.Feedforward_policy.forward(observations, deterministic=True)

        a_fb = actions - a_ff  # 反馈控制项
        z = torch.cat([e, a_fb], dim=-1)  # [B, D]

        # 使用 zᵀ H z 计算 Q 值
        H, _ = self.get_H_and_K()
        q_values = torch.einsum("bi,ij,bj->b", z, H, z).unsqueeze(-1)  # [B, 1]

        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)  # [B, N]

        return q_values

    def get_feedback_action(
            self,
            observations: torch.Tensor,
            repeat: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成反馈动作并计算 Q 值，使用 zᵀHz 方式代替 z_vec @ h，提升效率。
        """
        H, K = self.get_H_and_K()

        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)  # [B, repeat, obs_dim]
            B, N, _ = observations.shape

            x_cur = observations[:, :, :self.state_dim // 2]
            x_tar = observations[:, :, self.state_dim // 2:]

            with torch.no_grad():
                phi_cur = self.Koopman_encoder.encode(x_cur)  # [B, N, Nkoop]
                phi_tar = self.Koopman_encoder.encode(x_tar)

            e = phi_cur - phi_tar  # [B, N, Nkoop]
            a_fb = -torch.einsum("bij,jk->bik", e, K.T).detach()  # [B, N, A]

            z = torch.cat([e, a_fb], dim=-1)  # [B, N, D]
            z_flat = z.reshape(-1, z.shape[-1])  # [B*N, D]
            q_flat = torch.einsum("bi,ij,bj->b", z_flat, H, z_flat)  # [B*N]
            q_values = q_flat.view(B, N)  # [B, N]

        else:
            x_cur = observations[:, :self.state_dim // 2]
            x_tar = observations[:, self.state_dim // 2:]

            with torch.no_grad():
                phi_cur = self.Koopman_encoder.encode(x_cur)
                phi_tar = self.Koopman_encoder.encode(x_tar)

            e = phi_cur - phi_tar
            a_fb = -torch.matmul(e, K.T).detach()
            z = torch.cat([e, a_fb], dim=-1)  # [B, D]
            q_values = torch.einsum("bi,ij,bj->b", z, H, z).unsqueeze(-1)  # [B, 1]

        return a_fb.detach(), q_values

    def compute_H_and_K(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回当前 Q 函数对应的对称矩阵 H 和最优反馈控制律 K，使 a_fb = -K e 最小化代价。
        """

        # start_cql = time.perf_counter()

        D = self.Nkoopman + self.action_dim
        H = torch.zeros((D, D), device=self.h.device, dtype=self.h.dtype)
        # === 解码 self.h 成 H 矩阵 ===
        # 获取上三角索引
        iu = torch.triu_indices(D, D)
        # 拷贝一份 self.h 以便处理除2
        h_vec = self.h_cost.squeeze(-1).clone()
        # 对非对角项除以 2（注意：对角线不动）
        diag_mask = iu[0] == iu[1]
        h_vec[~diag_mask] = h_vec[~diag_mask] / 2
        # 赋值上三角
        H[iu[0], iu[1]] = h_vec
        # 对称填充下三角
        H = H + H.T - torch.diag(H.diagonal())

        # end_cql = time.perf_counter()
        # print(f"🕒 solve_K 总耗时: {end_cql - start_cql:.4f} 秒")
        # === 提取分块 ===
        H_ae = H[self.Nkoopman:, :self.Nkoopman]  # A × N
        H_aa = H[self.Nkoopman:, self.Nkoopman:]  # A × A

        # === 求解最优反馈律 K ===
        try:
            K = torch.linalg.solve(H_aa, H_ae)
        except RuntimeError:
            print("⚠️ H_aa 奇异，使用伪逆求解 K")
            K = torch.linalg.pinv(H_aa) @ H_ae

        return H, K.detach()

    def get_H_and_K(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cached_K is not None and self._cached_H is not None:
            return self._cached_H, self._cached_K
        H, K = self.compute_H_and_K()
        self._cached_H = H
        self._cached_K = K
        return self._cached_H, self._cached_K

    def reset_cache(self):
        self._cached_H = None
        self._cached_K = None

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        # 转换成 [1, obs_dim] 的张量
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action, _ = self.critic.get_feedback_action(state_tensor)
        return action.cpu().numpy().flatten()


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class KORL:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        critic_1_kop,
        critic_1_kop_optimizer,
        # critic_2_kop,
        # critic_2_kop_optimizer,
        actor,
        actor_optimizer,
        Koopman_encoder,
        k_optimizer: torch.optim.Optimizer,  # ✅ 显式要求传入
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
        reg_coeff_fb: float = 0,
        linear_coeff: float = 0,
        actor_update_num: int = 1,
        encode_update_num: int = 1,
        Q_mat: torch.Tensor = None,
        R_mat: torch.Tensor = None
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

        self.reg_coeff_fb = reg_coeff_fb
        self.linear_coeff = linear_coeff
        self.Koopman_encoder = Koopman_encoder
        self.K_optimizer = k_optimizer  # ✅ 直接使用外部传入的优化器
        self.actor_update_num = actor_update_num
        self.encode_update_num = encode_update_num

        self.critic_1_kop = critic_1_kop
        # self.critic_2_kop = critic_2_kop
        self.critic_1_kop_optimizer = critic_1_kop_optimizer
        # self.critic_2_kop_optimizer = critic_2_kop_optimizer
        self.target_critic_1_kop = deepcopy(self.critic_1_kop).to(device)
        # self.target_critic_2_kop = deepcopy(self.critic_2_kop).to(device)

        self.log_alpha_kop_prime = Scalar(1.0)
        self.alpha_kop_prime_optimizer = torch.optim.Adam(
            self.log_alpha_kop_prime.parameters(),
            lr=self.qf_lr*2,
        )

        self.Q_mat = Q_mat
        self.R_mat = R_mat

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)
        soft_update(self.target_critic_1_kop, self.critic_1_kop, soft_target_update_rate)
        # soft_update(self.target_critic_2_kop, self.critic_2_kop, soft_target_update_rate)
        self.target_critic_1_kop.reset_cache()
        # self.target_critic_2_kop.reset_cache()


    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(0, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def _koopman_q_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # start_cql = time.perf_counter()

        # Koopman-informed Q-value predictions (negative because they represent costs)
        q1_kop_pred = -self.critic_1_kop(observations, actions)
        # q2_kop_pred = -self.critic_2_kop(observations, actions)

        with torch.no_grad():
            a_fb1, q1 = self.target_critic_1_kop.get_feedback_action(next_observations)
            # a_fb2, q2 = self.target_critic_2_kop.get_feedback_action(next_observations)
            # target_q_kop = torch.min(-q1, -q2)
            target_q_kop = -q1
        # Bellman target
        td_target_kop = rewards + (1.0 - dones) * self.discount * target_q_kop.detach()
        qf1_loss_kop = F.mse_loss(q1_kop_pred, td_target_kop)
        # qf2_loss_kop = F.mse_loss(q2_kop_pred, td_target_kop)

        # CQL regularization
        batch_size, action_dim = actions.shape
        cql_rand_actions = torch.empty(
            (batch_size, self.cql_n_actions, action_dim),
            device=actions.device
        ).uniform_(0, 1)

        # cql_cur_actions = self.actor(observations, repeat=self.cql_n_actions)[0].detach()
        # cql_next_actions = self.actor(next_observations, repeat=self.cql_n_actions)[0].detach()
        # Negative Q-values since they represent costs
        q1_rand = -self.critic_1_kop(observations, cql_rand_actions)
        # q2_rand = -self.critic_2_kop(observations, cql_rand_actions)
        # q1_cur = -self.critic_1_kop(observations, cql_cur_actions)
        # q2_cur = -self.critic_2_kop(observations, cql_cur_actions)
        # q1_next = -self.critic_1_kop(observations, cql_next_actions)
        # q2_next = -self.critic_2_kop(observations, cql_next_actions)

        # start_fb = time.perf_counter()

        _, q1_cur = self.critic_1_kop.get_feedback_action(observations, repeat=self.cql_n_actions)
        # _, q2_cur = self.critic_2_kop.get_feedback_action(observations, repeat=self.cql_n_actions)
        _, q1_next = self.critic_1_kop.get_feedback_action(next_observations, repeat=self.cql_n_actions)
        # _, q2_next = self.critic_2_kop.get_feedback_action(next_observations, repeat=self.cql_n_actions)

        # end_fb = time.perf_counter()
        # print(f"🕒 get_feedback_action 耗时: {end_fb - start_fb:.4f} 秒")

        q1_cur = -q1_cur
        # q2_cur = -q2_cur
        q1_next = -q1_next
        # q2_next = -q2_next

        # logsumexp over all
        q1_cat = torch.cat([q1_rand, q1_kop_pred, q1_next, q1_cur], dim=1)
        # q2_cat = torch.cat([q2_rand, q2_kop_pred, q2_next, q2_cur], dim=1)

        cql_qf1_ood = torch.logsumexp(q1_cat / self.cql_temp, dim=1) * self.cql_temp
        # cql_qf2_ood = torch.logsumexp(q2_cat / self.cql_temp, dim=1) * self.cql_temp

        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_kop_pred,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max
        ).mean()
        # cql_qf2_diff = torch.clamp(
        #     cql_qf2_ood - q2_kop_pred,
        #     self.cql_clip_diff_min,
        #     self.cql_clip_diff_max
        # ).mean()

        if self.cql_lagrange:
            alpha_kop_prime = torch.clamp(
                torch.exp(self.log_alpha_kop_prime()), min=0.0, max=1000000.0
            )

            cql_min_qf1_loss_detached = (
                    alpha_kop_prime
                    * self.cql_alpha
                    * (cql_qf1_diff.detach() - self.cql_target_action_gap)
            )
            # cql_min_qf2_loss_detached = (
            #         alpha_kop_prime
            #         * self.cql_alpha
            #         * (cql_qf2_diff.detach() - self.cql_target_action_gap)
            # )

            self.alpha_kop_prime_optimizer.zero_grad()
            # alpha_kop_prime_loss = (-cql_min_qf1_loss_detached - cql_min_qf2_loss_detached) * 0.5
            alpha_kop_prime_loss = (-cql_min_qf1_loss_detached) * 1
            alpha_kop_prime_loss.backward(retain_graph=True)
            self.alpha_kop_prime_optimizer.step()

            cql_min_qf1_loss = (
                    alpha_kop_prime
                    * self.cql_alpha
                    * (cql_qf1_diff - self.cql_target_action_gap)
            )
            # cql_min_qf2_loss = (
            #         alpha_kop_prime
            #         * self.cql_alpha
            #         * (cql_qf2_diff - self.cql_target_action_gap)
            # )

        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            # cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_kop_prime_loss = observations.new_tensor(0.0)
            alpha_kop_prime = observations.new_tensor(0.0)

        # qf_loss_kop = qf1_loss_kop + qf2_loss_kop + cql_min_qf1_loss + cql_min_qf2_loss
        qf_loss_kop = qf1_loss_kop + cql_min_qf1_loss

        # Logging
        log_dict.update({
            'kop_qf1_loss': qf1_loss_kop.item(),
            # 'kop_qf2_loss': qf2_loss_kop.item(),
            'kop_q1_mean': q1_kop_pred.mean().item(),
            # 'kop_q2_mean': q2_kop_pred.mean().item(),
            'kop_target_q': target_q_kop.mean().item(),
            'kop_cql_qf1_diff': cql_qf1_diff.item(),
            # 'kop_cql_qf2_diff': cql_qf2_diff.item(),
            'kop_cql_min_qf1_loss': cql_min_qf1_loss.item(),
            # 'kop_cql_min_qf2_loss': cql_min_qf2_loss.item(),
            'alpha_kop_prime': alpha_kop_prime.item(),
            'alpha_kop_prime_loss': alpha_kop_prime_loss.item(),
        })
        # end_cql = time.perf_counter()
        # print(f"🕒 get_feedback_action (CQL) 总耗时: {end_cql - start_cql:.4f} 秒")

        return qf_loss_kop, alpha_kop_prime, alpha_kop_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        start_time = time.time()  # ⏱ 开始打点

        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        self.total_it += 1

        # 初始化异步 actor 机制状态
        if not hasattr(self, "_update_phase"):
            self._update_phase = "encode"
            self._phase_step = 0

        log_dict = {}

        if self._update_phase == "encode":
            async_linear_loss = self.update_Koopman_encoder(observations, actions, next_observations)
            log_dict["async_linear_loss"] = async_linear_loss.item()

            self._phase_step += 1
            if self._phase_step >= self.encode_update_num:
                self._update_phase = "actor"
                self._phase_step = 0

        elif self._update_phase == "actor":
            new_actions, log_pi = self.actor(observations)

            alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

            policy_loss = self._policy_loss(
                observations, actions, new_actions, alpha, log_pi
            )
            # l2_reg = torch.norm(self.actor.cur_phi_layer.weight, p=2)
            # policy_loss += self.reg_coeff_fb * l2_reg

            log_dict.update(dict(
                log_pi=log_pi.mean().item(),
                policy_loss=policy_loss.item(),
                alpha_loss=alpha_loss.item(),
                alpha=alpha.item(),
            ))

            qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
                observations, actions, next_observations, rewards, dones, alpha, log_dict
            )

            rewards_fb = -self.compute_cost_from_observations(observations, actions)

            qf_fb_loss, alpha_kop_prime, alpha_kop_prime_loss = self._koopman_q_loss(
                observations, actions, next_observations, rewards_fb, dones, log_dict
            )

            if self.use_automatic_entropy_tuning:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()
            qf_loss.backward(retain_graph=True)
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()

            self.critic_1_kop_optimizer.zero_grad()
            # self.critic_2_kop_optimizer.zero_grad()
            qf_fb_loss.backward(retain_graph=True)
            self.critic_1_kop_optimizer.step()
            # self.critic_2_kop_optimizer.step()
            self.critic_1_kop.reset_cache()
            # self.critic_2_kop.reset_cache()

            if self.total_it % self.target_update_period == 0:
                self.update_target_network(self.soft_target_update_rate)

            self._phase_step += 1
            if self._phase_step >= self.actor_update_num:
                self._update_phase = "encode"
                self._phase_step = 0

        # # ⏱ 打印或记录时长
        duration = time.time() - start_time
        log_dict["train_step_time"] = duration

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            # 基础 SAC / CQL 结构
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),

            # 新增 Koopman + kop critics
            "critic1_kop": self.critic_1_kop.state_dict(),
            # "critic2_kop": self.critic_2_kop.state_dict(),
            "critic1_kop_target": self.target_critic_1_kop.state_dict(),
            # "critic2_kop_target": self.target_critic_2_kop.state_dict(),
            "critic_1_kop_optimizer": self.critic_1_kop_optimizer.state_dict(),
            # "critic_2_kop_optimizer": self.critic_2_kop_optimizer.state_dict(),
            "koopman_encoder": self.Koopman_encoder.state_dict(),
            "koopman_optimizer": self.K_optimizer.state_dict(),
            "kop_log_alpha": self.log_alpha_kop_prime,
            "kop_log_alpha_optim": self.alpha_kop_prime_optimizer.state_dict(),

            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic_1.load_state_dict(state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict["critic2"])
        self.target_critic_1.load_state_dict(state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict["critic2_target"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(state_dict["sac_log_alpha_optim"])
        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(state_dict["cql_log_alpha_optim"])

        # 加载 Koopman + kop critics
        self.critic_1_kop.load_state_dict(state_dict["critic1_kop"])
        # self.critic_2_kop.load_state_dict(state_dict["critic2_kop"])
        self.target_critic_1_kop.load_state_dict(state_dict["critic1_kop_target"])
        # self.target_critic_2_kop.load_state_dict(state_dict["critic2_kop_target"])
        self.critic_1_kop_optimizer.load_state_dict(state_dict["critic_1_kop_optimizer"])
        # self.critic_2_kop_optimizer.load_state_dict(state_dict["critic_2_kop_optimizer"])
        self.Koopman_encoder.load_state_dict(state_dict["koopman_encoder"])
        self.K_optimizer.load_state_dict(state_dict["koopman_optimizer"])
        self.log_alpha_kop_prime = state_dict["kop_log_alpha"]
        self.alpha_kop_prime_optimizer.load_state_dict(state_dict["kop_log_alpha_optim"])

        self.total_it = state_dict["total_it"]

    def update_Koopman_encoder(
            self,
            observations: torch.Tensor,
            action: torch.Tensor,
            next_observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        更新 KoopmanEncoder 中的 encode_net、A、B，使其拟合 Koopman 动力学：
            phi(x_next) ≈ A phi(x) + B u
        """
        state_dim = observations.shape[1] // 2
        x_cur = observations[:, :state_dim]
        x_next = next_observations[:, :state_dim]

        phi_x_cur = self.Koopman_encoder.encode(x_cur)
        phi_x_next = self.Koopman_encoder.encode(x_next)

        A_mat = self.Koopman_encoder.form_A_from_eigenvalues()
        B_mat = self.Koopman_encoder.B

        pred_phi_x_next = phi_x_cur @ A_mat + action @ B_mat
        loss = nn.functional.mse_loss(pred_phi_x_next, phi_x_next)

        self.K_optimizer.zero_grad()
        loss.backward()
        self.K_optimizer.step()

        return loss

    def compute_cost_from_observations(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor
    ) -> torch.Tensor:
        """
        用于从 observation 和 action 构造 cost = e^T Q e + u^T R u。

        参数:
            - observations: [B, obs_dim]，前半是 x_cur，后半是 x_tar
            - actions: [B, act_dim]
            - Koopman_encoder: Koopman encoder 模块，包含 .encode() 方法
            - Q_mat: [Nkoopman, Nkoopman] Koopman 状态的权重矩阵
            - R_mat: [act_dim, act_dim] 动作的权重矩阵

        返回:
            - cost: [B]，每个样本的标量 cost 值
        """
        with torch.no_grad():
            state_dim = observations.shape[1] // 2
            x_cur = observations[:, :state_dim]
            x_tar = observations[:, state_dim:]

            phi_cur = self.Koopman_encoder.encode(x_cur)
            phi_tar = self.Koopman_encoder.encode(x_tar)
            e = phi_cur - phi_tar

            a_ff, _ = self.actor.forward(observations, deterministic=True)
            a_fb = actions - a_ff

            # e^T Q e 和 u^T R u 分别计算
            cost_state = torch.einsum("bi,ij,bj->b", e, self.Q_mat, e)
            cost_action = torch.einsum("bi,ij,bj->b", a_fb, self.R_mat, a_fb)

            total_cost = cost_state + cost_action  # [B]
            return total_cost.reshape(-1, 1)


@pyrallis.wrap()
def train(config: TrainConfig):
    # torch.autograd.set_detect_anomaly(True)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)


    # 加载 dataset
    dataset_mat = scipy.io.loadmat(config.dataset_path)
    # 提取变量
    rewards = dataset_mat['reward']  # (N, 1)
    states = dataset_mat['state']  # (N, state_dim)
    next_states = dataset_mat['state_next']  # (N, state_dim)
    actions = dataset_mat['u']  # (N, action_dim)
    # 获取状态维度（state_dim）
    state_dim = states.shape[1]
    # 获取动作维度（action_dim）
    action_dim = actions.shape[1]

    # 检查形状
    N = rewards.shape[0]
    assert states.shape[0] == N and next_states.shape[0] == N and actions.shape[0] == N
    # 创建终止标志（全为 0，表示没有 episode 终止）
    terminals = np.zeros((N, 1), dtype=np.float32)
    # 构造符合 d4rl 格式的 dataset 字典
    dataset = {
        'observations': states.astype(np.float32),
        'actions': actions.astype(np.float32),
        'rewards': rewards.astype(np.float32),
        'next_observations': next_states.astype(np.float32),
        'terminals': terminals
    }

    if config.normalize_reward:
        # modify_reward(dataset, config.env)
        rewards_mean, rewards_std = modify_reward(dataset)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    # max_action = float(env.action_space.high[0])
    max_action = 1

    # Set seeds
    seed = config.seed
    set_seed(seed)


    # -----------------------------
    # Define Koopman Encoder
    # -----------------------------
    Koopman_encoder = KoopmanEncoder(
        encode_layers=config.koopman_encode_layers,  # e.g., [12, 256, 256, 24]
        u_dim=action_dim
    ).to(config.device)

    k_optimizer = torch.optim.Adam(
        Koopman_encoder.parameters(),  # ✅ 更新 A, B, encode_net
        lr=config.k_lr
    )

    # -----------------------------
    # Define actor
    # -----------------------------
    actor = KoopmanFeedforwardPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        Koopman_encoder=Koopman_encoder,  # ✅ 注入 encoder
        log_std_offset=0,
        log_std_multiplier=config.policy_log_std_multiplier,
        # orthogonal_init=config.orthogonal_init,
    ).to(config.device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.policy_lr)

    # -----------------------------
    # Define critic networks
    # -----------------------------
    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)

    critic_2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)

    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.qf_lr)

    critic_1_kop = KoopmanInformedQFunction(
        Koopman_encoder,
        actor,
        state_dim,
        action_dim,
    ).to(config.device)

    # critic_2_kop = KoopmanInformedQFunction(
    #     Koopman_encoder,
    #     actor,
    #     state_dim,
    #     action_dim,
    # ).to(config.device)

    # critic_1_kop_optimizer = torch.optim.Adam(critic_1_kop.parameters(), lr=config.qf_lr*2)
    # critic_2_kop_optimizer = torch.optim.Adam(critic_2_kop.parameters(), lr=config.qf_lr)
    critic_1_kop_optimizer = torch.optim.Adam(
        critic_1_kop.parameters(),
        lr=config.qf_lr * 2,
        weight_decay=1e-5  # l2 reg
    )

    Nkoopman = config.koopman_encode_layers[0] + config.koopman_encode_layers[-1]
    Q_diag = 0.5 * np.ones(Nkoopman) if len(config.Q_diag) == 0 else np.array(config.Q_diag)
    R_diag = 10 * np.ones(action_dim) if len(config.R_diag) == 0 else np.array(config.R_diag)
    Q_mat = torch.diag(torch.tensor(Q_diag, dtype=torch.float32)).to(config.device)
    R_mat = torch.diag(torch.tensor(R_diag, dtype=torch.float32)).to(config.device)
    # print(R_mat)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(action_dim).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        # Koopman
        "reg_coeff_fb": config.reg_coeff_fb,
        "k_optimizer": k_optimizer,
        "actor_update_num": config.actor_update_num,
        "encode_update_num": config.encode_update_num,
        "Koopman_encoder": Koopman_encoder,
        "critic_1_kop": critic_1_kop,
        # "critic_2_kop": critic_2_kop,
        "critic_1_kop_optimizer": critic_1_kop_optimizer,
        # "critic_2_kop_optimizer": critic_2_kop_optimizer,
        'Q_mat': Q_mat,
        'R_mat': R_mat
    }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = KORL(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")

            # ✅ 线性性评估
            linearity_loss = eval_encoder_linear(
                Koopman_encoder=Koopman_encoder,
                dataset=dataset,
                device=config.device,
            )
            print(f"Koopman_linear_loss (MSE): {linearity_loss:.6f}")
            print("---------------------------------------")

            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t+1}.pt"),
                )

            wandb.log({
                # "HPN_eval_score": eval_score,
                "Koopman_linear_loss": linearity_loss
            },
                step=trainer.total_it,
            )


if __name__ == "__main__":
    train()
