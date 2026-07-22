from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from soft_vla.runtime.smolvla_async_runtime import (
    _apply_sigmoid_to_gripper_action_chunk,
    _postprocess_gripper_sequence,
    _resolve_sigmoid_bounded_gripper,
)


class RuntimeGripperPostprocessTest(unittest.TestCase):
    def test_sigmoid_contract_is_detected_from_training_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint = run_dir / "checkpoints" / "020000" / "pretrained_model"
            checkpoint.mkdir(parents=True)
            (run_dir / "train_summary.json").write_text(
                json.dumps({"gripper_action_head": {"sigmoid_bounded": True}}),
                encoding="utf-8",
            )

            enabled, source = _resolve_sigmoid_bounded_gripper(checkpoint, override=None)

            self.assertTrue(enabled)
            self.assertEqual(source, str(run_dir / "train_summary.json"))

    def test_explicit_override_wins_over_checkpoint_metadata(self):
        enabled, source = _resolve_sigmoid_bounded_gripper(Path("missing"), override=False)
        self.assertFalse(enabled)
        self.assertEqual(source, "cli_override")

    def test_missing_checkpoint_metadata_preserves_legacy_behavior(self):
        enabled, source = _resolve_sigmoid_bounded_gripper(Path("missing"), override=None)
        self.assertFalse(enabled)
        self.assertEqual(source, "checkpoint_metadata_absent")

    def test_sigmoid_bounded_gripper_only_changes_action_index_six(self):
        actions = np.zeros((1, 3, 19), dtype=np.float32)
        actions[0, :, 6] = np.asarray([-2.0, 0.0, 2.0], dtype=np.float32)
        bounded = _apply_sigmoid_to_gripper_action_chunk(actions)

        np.testing.assert_allclose(
            bounded[0, :, 6],
            1.0 / (1.0 + np.exp(-np.asarray([-2.0, 0.0, 2.0], dtype=np.float32))),
        )
        np.testing.assert_array_equal(bounded[..., :6], actions[..., :6])
        np.testing.assert_array_equal(bounded[..., 7:], actions[..., 7:])
        np.testing.assert_array_equal(actions[0, :, 6], [-2.0, 0.0, 2.0])

    def test_sigmoid_bounded_gripper_rejects_missing_dimension(self):
        with self.assertRaises(ValueError):
            _apply_sigmoid_to_gripper_action_chunk(np.zeros((2, 6), dtype=np.float32))

    def test_thresholds_keep_previous_state_in_middle_band(self):
        raw = np.asarray([0.5, 0.81, 0.4, 0.19, 0.3, 0.9], dtype=np.float32)
        out = _postprocess_gripper_sequence(
            raw,
            previous_gripper=0.0,
            close_threshold=0.2,
            open_threshold=0.8,
        )
        np.testing.assert_array_equal(out, np.asarray([0, 1, 1, 0, 0, 1], dtype=np.float32))

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            _postprocess_gripper_sequence(
                np.asarray([0.5], dtype=np.float32),
                previous_gripper=1.0,
                close_threshold=0.8,
                open_threshold=0.2,
            )

    def test_legacy_logits_allow_thresholds_outside_unit_interval(self):
        raw = np.asarray([-0.2, 0.5, 1.2], dtype=np.float32)
        out = _postprocess_gripper_sequence(
            raw,
            previous_gripper=1.0,
            close_threshold=-0.1,
            open_threshold=1.1,
        )
        np.testing.assert_array_equal(out, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
