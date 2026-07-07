from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn


def build_mlp(layer_sizes: list[int]) -> nn.Sequential:
    layers = OrderedDict()
    for idx, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers[f"linear_{idx}"] = nn.Linear(in_dim, out_dim)
        if idx != len(layer_sizes) - 2:
            layers[f"relu_{idx}"] = nn.ReLU()
    return nn.Sequential(layers)


class FullAHistoryKoopmanNetwork(nn.Module):
    """History-context Deep Koopman model with full trainable A.

    Row-vector convention is preserved from the legacy project:

    z_next = z @ A + u @ B + bias
    """

    def __init__(
        self,
        context_dim: int,
        n_state: int,
        u_dim: int,
        encode_dim: int,
        hidden_sizes: list[int],
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.n_state = int(n_state)
        self.u_dim = int(u_dim)
        self.encode_dim = int(encode_dim)
        self.n_koopman = self.n_state + self.encode_dim
        self.hidden_sizes = [int(v) for v in hidden_sizes]

        self.encoder = build_mlp([self.context_dim] + self.hidden_sizes + [self.encode_dim])
        self.A = nn.Parameter(torch.eye(self.n_koopman) + 0.001 * torch.randn(self.n_koopman, self.n_koopman))
        self.B = nn.Parameter(0.01 * torch.randn(self.u_dim, self.n_koopman))
        self.bias = nn.Parameter(torch.zeros(self.n_koopman))

    def encode_only(self, context: torch.Tensor) -> torch.Tensor:
        return self.encoder(context.float())

    def encode(self, current_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.cat([current_state.float(), self.encode_only(context)], dim=-1)

    def forward(self, lifted_state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lifted_state, self.A) + torch.matmul(control.float(), self.B) + self.bias


@dataclass(frozen=True)
class FullAHistoryLossWeights:
    koopman: float = 10.0
    pred: float = 1.0
    stability: float = 0.01
    std: float = 0.1
    identity: float = 1e-4
    svd: float = 0.0
    augment: float = 0.0


def compute_std_loss(
    phi: torch.Tensor,
    target_std: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    phi_std = phi.reshape(-1, phi.shape[-1]).std(dim=0, unbiased=False)
    std_loss = torch.relu(float(target_std) - phi_std).pow(2).mean()
    return std_loss, {
        "std_loss": std_loss,
        "latent_std_min": phi_std.min(),
        "latent_std_mean": phi_std.mean(),
        "latent_std_max": phi_std.max(),
    }


def compute_stability_loss(
    model: FullAHistoryKoopmanNetwork,
    spectral_radius_limit: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eig_abs = torch.linalg.eigvals(model.A).abs()
    stability_loss = torch.relu(eig_abs - float(spectral_radius_limit)).pow(2).mean()
    return stability_loss, {
        "stability_loss": stability_loss,
        "spectral_radius": eig_abs.max().real,
        "eig_abs_mean": eig_abs.mean().real,
        "eig_abs_max": eig_abs.max().real,
    }


def compute_identity_loss(model: FullAHistoryKoopmanNetwork, mse_loss: nn.Module) -> torch.Tensor:
    eye = torch.eye(model.n_koopman, dtype=model.A.dtype, device=model.A.device)
    return mse_loss(model.A, eye)


def compute_svd_loss(model: FullAHistoryKoopmanNetwork, min_singular_value: float) -> torch.Tensor:
    a_dim = model.n_koopman
    temp = model.B.T
    controllability = model.A.new_zeros((a_dim, a_dim * model.u_dim))
    for idx in range(a_dim):
        controllability[:, idx * model.u_dim : (idx + 1) * model.u_dim] = temp
        temp = torch.mm(model.A.T, temp)
    singular_values = torch.linalg.svdvals(controllability)
    return torch.relu(float(min_singular_value) - singular_values).pow(2).mean()


def define_fullA_history_loss(
    context_sequence: torch.Tensor,
    current_state_sequence: torch.Tensor,
    control_sequence: torch.Tensor,
    state_target_sequence: torch.Tensor,
    net: FullAHistoryKoopmanNetwork,
    mse_loss: nn.Module,
    gamma: float,
    weights: FullAHistoryLossWeights,
    spectral_radius_limit: float,
    target_std: float,
    svd_min_singular_value: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Legacy-style open-loop Koopman loss with history-context encoder.

    context_sequence: [batch, K + 1, context_dim]
    current_state_sequence: [batch, K + 1, n_state]
    control_sequence: [batch, K, u_dim]
    state_target_sequence: [batch, K, n_state]

    The linear loss keeps the original full lifted-state alignment:
    MSE(predicted z_next, [true x_next, encoder(true context_next)]).
    """

    ksteps = control_sequence.shape[1]
    z_current = net.encode(current_state_sequence[:, 0], context_sequence[:, 0])
    linear_loss = context_sequence.new_zeros(())
    pred_loss = context_sequence.new_zeros(())
    beta = 1.0
    beta_sum = 0.0

    for step in range(ksteps):
        z_current = net.forward(z_current, control_sequence[:, step])
        true_next_state = state_target_sequence[:, step]
        true_next_phi = net.encode_only(context_sequence[:, step + 1])
        z_target = torch.cat([true_next_state, true_next_phi], dim=-1)
        linear_loss = linear_loss + beta * mse_loss(z_current, z_target)
        pred_loss = pred_loss + beta * mse_loss(z_current[:, : net.n_state], true_next_state)
        beta_sum += beta
        beta *= gamma

    linear_loss = linear_loss / beta_sum
    pred_loss = pred_loss / beta_sum

    all_true_contexts = context_sequence.reshape(-1, context_sequence.shape[-1])
    all_true_phi = net.encode_only(all_true_contexts)
    std_loss, std_stats = compute_std_loss(all_true_phi, target_std)
    stability_loss, stability_stats = compute_stability_loss(net, spectral_radius_limit)
    identity_loss = compute_identity_loss(net, mse_loss)
    if weights.svd > 0:
        svd_loss = compute_svd_loss(net, svd_min_singular_value)
    else:
        svd_loss = context_sequence.new_zeros(())
    augment_loss = context_sequence.new_zeros(())

    loss = (
        weights.koopman * linear_loss
        + weights.pred * pred_loss
        + weights.stability * stability_loss
        + weights.std * std_loss
        + weights.identity * identity_loss
        + weights.svd * svd_loss
        + weights.augment * augment_loss
    )
    components = {
        "loss": loss,
        "linear_loss": linear_loss,
        "pred_loss": pred_loss,
        "stability_loss": stability_loss,
        "std_loss": std_loss,
        "identity_loss": identity_loss,
        "svd_loss": svd_loss,
        "augment_loss": augment_loss,
        **std_stats,
        **stability_stats,
        "A_norm": net.A.norm(),
        "B_norm": net.B.norm(),
        "A_latent_to_state_norm": net.A[net.n_state :, : net.n_state].norm(),
        "A_state_to_latent_norm": net.A[: net.n_state, net.n_state :].norm(),
    }
    return loss, components
