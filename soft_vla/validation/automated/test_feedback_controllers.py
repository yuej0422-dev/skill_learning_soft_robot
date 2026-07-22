from __future__ import annotations

import unittest

import numpy as np

from soft_vla.motion_control.feedback_controllers import (
    IntegralFeedbackConfig,
    IntegralFeedbackController,
    build_augmented_system,
    make_integral_lqr_q_weights,
    solve_discrete_are_iterative,
)


class FeedbackControllerTest(unittest.TestCase):
    def test_augmented_system_dimensions(self):
        A = np.eye(24)
        B = np.zeros((12, 24))
        C = np.concatenate([np.eye(6), np.zeros((6, 18))], axis=1)
        At, Bt = build_augmented_system(A, B, C, 0.02)
        self.assertEqual(At.shape, (30, 30))
        self.assertEqual(Bt.shape, (30, 12))

    def test_integral_state_updates_and_clips(self):
        C = np.concatenate([np.eye(6), np.zeros((6, 18))], axis=1)
        K = np.zeros((12, 30))
        K[0, 24] = 1.0
        controller = IntegralFeedbackController(
            K=K,
            C=C,
            config=IntegralFeedbackConfig(max_integral_error=0.01, feedback_gain_scale=1.0),
        )
        err = np.zeros(24)
        err[0] = 10.0
        out = controller.predict(err)
        self.assertAlmostEqual(float(controller.q[0]), 0.01)
        self.assertAlmostEqual(float(out[0]), -0.01)

    def test_q_weight_layout(self):
        weights = make_integral_lqr_q_weights(
            n_koopman=24,
            ny=6,
            tcp6_weight=1.0,
            state_tail_weight=0.1,
            latent_weight=0.1,
            integral_weight=0.5,
        )
        self.assertEqual(weights.shape, (30,))
        self.assertTrue(np.allclose(weights[:6], 1.0))
        self.assertTrue(np.allclose(weights[6:24], 0.1))
        self.assertTrue(np.allclose(weights[24:], 0.5))

    def test_numpy_riccati_fallback_returns_finite_matrix(self):
        A = np.asarray([[1.0, 0.1], [0.0, 0.95]])
        B = np.asarray([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.eye(1)
        P = solve_discrete_are_iterative(A, B, Q, R, max_iterations=2000)
        self.assertEqual(P.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(P)))
        self.assertTrue(np.allclose(P, P.T))


if __name__ == "__main__":
    unittest.main()
