from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from .model_fullA_history import (
        FullAHistoryKoopmanNetwork,
        FullAHistoryLossWeights,
        compute_std_loss,
        define_fullA_history_loss,
    )
    from .train_fullA_history import (
        build_history_koopman_buffer,
        context_at,
        load_checkpoint,
        save_checkpoint,
    )
except ImportError:  # pragma: no cover
    from model_fullA_history import (
        FullAHistoryKoopmanNetwork,
        FullAHistoryLossWeights,
        compute_std_loss,
        define_fullA_history_loss,
    )
    from train_fullA_history import (
        build_history_koopman_buffer,
        context_at,
        load_checkpoint,
        save_checkpoint,
    )


def assert_close(values, expected) -> None:
    np.testing.assert_allclose(np.asarray(values, dtype=np.float32), np.asarray(expected, dtype=np.float32), atol=1e-6)


def make_model(context_dim: int = 240) -> FullAHistoryKoopmanNetwork:
    return FullAHistoryKoopmanNetwork(
        context_dim=context_dim,
        n_state=12,
        u_dim=12,
        encode_dim=12,
        hidden_sizes=[32],
    )


def test_history_window_timing() -> None:
    states = np.arange(70, dtype=np.float32).reshape(-1, 1)
    pressures = (100 + np.arange(70, dtype=np.float32)).reshape(-1, 1)
    context = context_at(states, pressures, t=10, history_steps=10)
    assert_close(context[:10], list(range(1, 11)))
    assert_close(context[10:], list(range(100, 110)))

    next_context = context_at(states, pressures, t=11, history_steps=10)
    assert_close(next_context[:10], list(range(2, 12)))
    assert_close(next_context[10:], list(range(101, 111)))

    contexts, current_states, controls, targets, stats = build_history_koopman_buffer(
        {0: (states, pressures)},
        [0],
        state_mean=np.zeros(1, dtype=np.float32),
        state_std=np.ones(1, dtype=np.float32),
        history_steps=10,
        ksteps=50,
    )
    assert contexts.shape == (10, 51, 20)
    assert current_states.shape == (10, 51, 1)
    assert controls.shape == (10, 50, 1)
    assert targets.shape == (10, 50, 1)
    assert stats["processed_frames"] == stats["original_frames"] == 70
    assert stats["upsample_factor"] == 1
    assert_close(contexts[0, 0, :10], list(range(1, 11)))
    assert_close(contexts[0, 0, 10:], list(range(100, 110)))
    assert_close(controls[0, 0, 0], 110)
    assert_close(contexts[0, 1, :10], list(range(2, 12)))
    assert_close(contexts[0, 1, 10:], list(range(101, 111)))
    assert_close(targets[0, 0, 0], current_states[0, 1, 0])


def test_no_cross_episode() -> None:
    states0 = np.arange(70, dtype=np.float32).reshape(-1, 1)
    pressures0 = (100 + np.arange(70, dtype=np.float32)).reshape(-1, 1)
    states1 = (1000 + np.arange(70, dtype=np.float32)).reshape(-1, 1)
    pressures1 = (2000 + np.arange(70, dtype=np.float32)).reshape(-1, 1)
    contexts, _, _, _, stats = build_history_koopman_buffer(
        {0: (states0, pressures0), 1: (states1, pressures1)},
        [0, 1],
        state_mean=np.zeros(1, dtype=np.float32),
        state_std=np.ones(1, dtype=np.float32),
        history_steps=10,
        ksteps=50,
    )
    assert stats["processed_frames"] == stats["original_frames"] == 140
    for sample in contexts:
        state_values = sample[:, :10]
        assert np.all(state_values < 100) or np.all(state_values > 1000)


def test_model_shapes() -> None:
    model = make_model()
    context = torch.randn(4, 240)
    state = torch.randn(4, 12)
    control = torch.randn(4, 12)
    phi = model.encode_only(context)
    z = model.encode(state, context)
    z_next = model(z, control)
    assert phi.shape == (4, 12)
    assert z.shape == (4, 24)
    assert z_next.shape == (4, 24)
    assert model.A.shape == (24, 24)
    assert model.B.shape == (12, 24)


def test_linear_loss_zero_and_positive() -> None:
    torch.manual_seed(0)
    model = make_model()
    with torch.no_grad():
        model.A.copy_(torch.eye(model.n_koopman))
        model.B.zero_()
        model.bias.zero_()
        for parameter in model.encoder.parameters():
            parameter.zero_()

    contexts = torch.zeros(2, 2, 240)
    states = torch.zeros(2, 2, 12)
    controls = torch.zeros(2, 1, 12)
    targets = torch.zeros(2, 1, 12)
    weights = FullAHistoryLossWeights(koopman=10, pred=1, stability=0, std=0, identity=0, svd=0, augment=0)
    _, components = define_fullA_history_loss(
        contexts,
        states,
        controls,
        targets,
        model,
        nn.MSELoss(),
        gamma=0.99,
        weights=weights,
        spectral_radius_limit=1.0,
        target_std=1.0,
    )
    assert float(components["linear_loss"]) == 0.0

    targets[:, 0, 0] = 1.0
    _, components = define_fullA_history_loss(
        contexts,
        states,
        controls,
        targets,
        model,
        nn.MSELoss(),
        gamma=0.99,
        weights=weights,
        spectral_radius_limit=1.0,
        target_std=1.0,
    )
    assert float(components["linear_loss"]) > 0.0


def test_fullA_gradients() -> None:
    torch.manual_seed(1)
    model = make_model()
    contexts = torch.randn(4, 3, 240)
    states = torch.randn(4, 3, 12)
    controls = torch.randn(4, 2, 12)
    targets = states[:, 1:].clone()
    loss, _ = define_fullA_history_loss(
        contexts,
        states,
        controls,
        targets,
        model,
        nn.MSELoss(),
        gamma=0.99,
        weights=FullAHistoryLossWeights(),
        spectral_radius_limit=1.0,
        target_std=1.0,
    )
    loss.backward()
    encoder_grad = sum(float(p.grad.detach().abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
    assert model.A.grad is not None and float(model.A.grad.abs().sum()) > 0
    assert model.B.grad is not None and float(model.B.grad.abs().sum()) > 0
    assert model.bias.grad is not None and float(model.bias.grad.abs().sum()) > 0
    assert encoder_grad > 0
    assert float(model.A.grad[model.n_state :, : model.n_state].abs().sum()) > 0


def test_target_std_loss() -> None:
    constant_phi = torch.zeros(16, 12)
    constant_loss, _ = compute_std_loss(constant_phi, target_std=1.0)
    varied_phi = torch.tensor([[-1.0] * 12, [1.0] * 12] * 8)
    varied_loss, stats = compute_std_loss(varied_phi, target_std=1.0)
    assert float(constant_loss) > 0.9
    assert float(varied_loss) < 1e-6
    assert abs(float(stats["latent_std_mean"]) - 1.0) < 1e-6


def test_checkpoint_roundtrip() -> None:
    torch.manual_seed(2)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metadata = {"history_steps": 10, "ksteps": 50, "target_std": 1.0}
    config = {"history_steps": 10, "ksteps": 50, "target_std": 1.0}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.pt"
        save_checkpoint(path, model, optimizer, epoch=7, best_val_loss=0.5, config=config, metadata=metadata)
        loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["optimizer_state_dict"]
    assert checkpoint["metadata"]["history_steps"] == 10
    assert checkpoint["metadata"]["ksteps"] == 50
    assert checkpoint["metadata"]["target_std"] == 1.0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), loaded.state_dict()[key].cpu())


def main() -> None:
    test_history_window_timing()
    test_no_cross_episode()
    test_model_shapes()
    test_linear_loss_zero_and_positive()
    test_fullA_gradients()
    test_target_std_loss()
    test_checkpoint_roundtrip()
    print("All Full-A history v2 tests passed.")


if __name__ == "__main__":
    main()
