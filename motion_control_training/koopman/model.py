from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class KoopmanNetwork(nn.Module):
    """Deep Koopman model adapted from the HPN reference script.

    The lifted state is ``z = [x, encoder(x)]``. The transition is linear in
    the lifted space with raw pressure controls: ``z_next = z A + u B``.
    """

    def __init__(self, encode_layers: list[int], n_koopman: int, u_dim: int) -> None:
        super().__init__()
        layers = OrderedDict()
        for layer_i, (in_dim, out_dim) in enumerate(zip(encode_layers[:-1], encode_layers[1:])):
            layers[f"linear_{layer_i}"] = nn.Linear(in_dim, out_dim)
            if layer_i != len(encode_layers) - 2:
                layers[f"relu_{layer_i}"] = nn.ReLU()
        self.encode_net = nn.Sequential(layers)
        self.encode_layers = [int(v) for v in encode_layers]
        self.n_koopman = int(n_koopman)
        self.u_dim = int(u_dim)
        self.num_real = int(np.mod(self.n_koopman, 2))
        self.num_complex_pair = int(self.n_koopman / 2)

        self.B = nn.Parameter(torch.empty(self.u_dim, self.n_koopman))
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        self.A = nn.Parameter(torch.normal(0.0, 0.01, size=(self.n_koopman,)))

    def encode_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_net(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def form_A_from_eigenvalues(self) -> torch.Tensor:
        dtype = self.A.dtype
        device = self.A.device
        temp_A = torch.zeros((self.n_koopman, self.n_koopman), dtype=dtype, device=device)
        for i in range(self.num_complex_pair):
            idx = 2 * i
            temp_A[idx : idx + 2, idx : idx + 2] = form_complex_conjugate_block(
                self.A[idx],
                self.A[idx + 1],
            )
        for i in range(self.num_real):
            idx = 2 * self.num_complex_pair + i
            temp_A[idx, idx] = self.A[idx]
        return temp_A

    def forward(self, lifted_state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lifted_state, self.form_A_from_eigenvalues()) + torch.matmul(control, self.B)


def form_complex_conjugate_block(real: torch.Tensor, imaginary: torch.Tensor) -> torch.Tensor:
    block = real.new_zeros((2, 2))
    block[0, 0] = real
    block[0, 1] = imaginary
    block[1, 0] = -imaginary
    block[1, 1] = real
    return block


@dataclass(frozen=True)
class KoopmanLossWeights:
    koopman: float = 10.0
    a_eig: float = 0.003
    svd: float = 0.003
    augment: float = 1.0
    pred: float = 1.0


def define_koopman_loss(
    data: torch.Tensor,
    net: KoopmanNetwork,
    mse_loss: nn.Module,
    u_dim: int,
    gamma: float,
    n_state: int,
    weights: KoopmanLossWeights = KoopmanLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reference Koopman objective.

    ``data`` has shape ``[batch, Ksteps + 1, u_dim + n_state]`` where controls
    are raw pressure values and states are already normalized.
    """

    _, steps, _ = data.shape
    device = data.device
    x_current = net.encode(data[:, 0, u_dim:])
    beta = 1.0
    beta_sum = 0.0

    linear_loss = torch.zeros((), dtype=data.dtype, device=device)
    a_eig_loss = torch.zeros((), dtype=data.dtype, device=device)
    svd_loss = torch.zeros((), dtype=data.dtype, device=device)
    augment_loss = torch.zeros((), dtype=data.dtype, device=device)
    pred_loss = torch.zeros((), dtype=data.dtype, device=device)

    for i in range(steps - 1):
        x_current = net.forward(x_current, data[:, i, :u_dim])
        beta_sum += beta

        pred_loss = pred_loss + beta * mse_loss(x_current[:, :n_state], data[:, i + 1, u_dim:])

        x_next_encode = net.encode(data[:, i + 1, u_dim:])
        linear_loss = linear_loss + beta * mse_loss(x_current, x_next_encode)

        beta *= gamma

        x_current_encoded = net.encode(x_current[:, :n_state])
        augment_loss = augment_loss + mse_loss(x_current_encoded, x_current)

    pred_loss = pred_loss / beta_sum
    linear_loss = linear_loss / beta_sum

    am = net.form_A_from_eigenvalues().T
    a_dim = net.n_koopman
    temp = net.B.T
    controllability_matrix = data.new_zeros((a_dim, a_dim * u_dim))
    for i in range(a_dim):
        controllability_matrix[:, i * u_dim : (i + 1) * u_dim] = temp
        temp = torch.mm(am, temp)
    singular_values = torch.linalg.svdvals(controllability_matrix)
    svd_margin = singular_values.abs() - 0.2
    for item in svd_margin[svd_margin < 0]:
        svd_loss = svd_loss + torch.norm(item, p=2)

    for i in range(net.num_complex_pair):
        idx = 2 * i
        a_eig_loss = a_eig_loss + torch.norm(net.A[idx].abs() - 0.5, p=2)
        a_eig_loss = a_eig_loss + torch.norm(net.A[idx + 1].abs(), p=2)
    for i in range(net.num_real):
        idx = 2 * net.num_complex_pair + i
        a_eig_loss = a_eig_loss + torch.norm(net.A[idx].abs() - 0.5, p=2)

    loss = (
        weights.koopman * linear_loss
        + weights.a_eig * a_eig_loss
        + weights.svd * svd_loss
        + weights.augment * augment_loss
        + weights.pred * pred_loss
    )
    components = {
        "loss": loss,
        "linear_loss": linear_loss,
        "a_eig_loss": a_eig_loss,
        "svd_loss": svd_loss,
        "augment_loss": augment_loss,
        "pred_loss": pred_loss,
    }
    return loss, components

