from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_koopman_pressure_data.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("collect_koopman_pressure_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def test_closed_gripper_packet_is_16_channels() -> None:
    norm16, physical16 = collector.build_pressure_packet(np.full(12, 0.5, dtype=np.float32))

    assert norm16.shape == (16,)
    assert physical16.shape == (16,)
    np.testing.assert_allclose(physical16[:12], 1.5)
    np.testing.assert_allclose(physical16[12:], [0.0, 3.0, 0.0, 0.0])
    np.testing.assert_allclose(norm16[12:], [0.0, 1.0, 0.0, 0.0])


def test_all_reference_excitation_terms_are_finite_and_bounded() -> None:
    rng = np.random.default_rng(7)
    parameters = collector.sample_excitation_parameters(0.8, rng)

    for term in range(8):
        values = collector.excitation_motion_norm12(
            term,
            step=37,
            sample_period_s=0.02,
            parameters=parameters,
            noise_lambda=0.2,
        )
        assert values.shape == (12,)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


def test_index_spec_supports_ranges_and_lists() -> None:
    assert collector.parse_index_spec("0:3,5", upper_bound=8) == [0, 1, 2, 5]


def test_existing_episode_count_requires_contiguous_files(tmp_path: Path) -> None:
    (tmp_path / "soft_robot_data_0000.mat").touch()
    (tmp_path / "soft_robot_data_0001.mat").touch()
    assert collector.existing_episode_count(tmp_path) == 2

    (tmp_path / "soft_robot_data_0003.mat").touch()
    try:
        collector.existing_episode_count(tmp_path)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous episode indices should be rejected")


def test_resume_config_ignores_serial_port_but_not_collection_plan() -> None:
    existing = {
        "format_version": 1,
        "serial": {"port": "/dev/ttyUSB0", "baudrate": 115200, "packet_channels": 16},
        "terms": [0],
    }
    requested = {
        "format_version": 1,
        "serial": {"port": "/dev/serial/by-id/device", "baudrate": 115200, "packet_channels": 16},
        "terms": [0],
    }
    assert collector.resume_config_errors(existing, requested) == []

    requested["terms"] = [0, 1]
    assert any(error.startswith("terms:") for error in collector.resume_config_errors(existing, requested))


def test_resume_allows_increasing_but_not_decreasing_repeats() -> None:
    existing = {"format_version": 1, "repeats": 1}
    requested = {"format_version": 1, "repeats": 3}
    assert collector.resume_config_errors(existing, requested) == []

    existing["repeats"] = 3
    requested["repeats"] = 2
    assert any("repeats cannot decrease" in error for error in collector.resume_config_errors(existing, requested))
