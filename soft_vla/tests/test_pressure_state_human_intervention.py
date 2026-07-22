from __future__ import annotations

import numpy as np

from soft_vla.runtime.smolvla_human_intervention_runtime import (
    HumanInterventionRuntimeConfig,
    _select_executed_pressure_action,
)


def test_human_intervention_executes_zero_pressure_delta_from_current_pressure() -> None:
    config = HumanInterventionRuntimeConfig(
        vla_action_mode="pressure_delta19",
        pressure_delta_scale=0.5,
    )
    current_pressure = np.linspace(0.1, 0.6, 12, dtype=np.float32)
    vla_delta = np.full(12, 0.2, dtype=np.float32)
    vla_feedforward = np.clip(current_pressure + 0.5 * vla_delta, 0.0, 1.0)

    executed_feedforward, executed_delta = _select_executed_pressure_action(
        config,
        action_source="human",
        latest_state={"motion_norm12": current_pressure.tolist()},
        vla_feedforward_pressure12=vla_feedforward,
        vla_pressure_delta12=vla_delta,
    )

    np.testing.assert_allclose(executed_feedforward, current_pressure)
    np.testing.assert_array_equal(executed_delta, np.zeros(12, dtype=np.float32))
    # Shadow outputs remain unchanged for logging/training diagnostics.
    np.testing.assert_allclose(vla_feedforward, current_pressure + 0.1)
    np.testing.assert_allclose(vla_delta, 0.2)


def test_vla_mode_still_executes_vla_pressure_delta() -> None:
    config = HumanInterventionRuntimeConfig(vla_action_mode="pressure_delta19")
    vla_delta = np.full(12, -0.05, dtype=np.float32)
    vla_feedforward = np.full(12, 0.4, dtype=np.float32)

    executed_feedforward, executed_delta = _select_executed_pressure_action(
        config,
        action_source="vla",
        latest_state={"motion_norm12": [0.7] * 12},
        vla_feedforward_pressure12=vla_feedforward,
        vla_pressure_delta12=vla_delta,
    )

    np.testing.assert_allclose(executed_feedforward, vla_feedforward)
    np.testing.assert_allclose(executed_delta, vla_delta)
