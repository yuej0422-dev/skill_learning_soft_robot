from __future__ import annotations

import unittest

import numpy as np

from soft_vla.runtime.smolvla_async_runtime import _postprocess_gripper_sequence


class RuntimeGripperPostprocessTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
