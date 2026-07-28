import unittest
from pathlib import Path

import numpy as np

from scripts.realworld.controllers import Mpc_controller


class UpstreamNavdpMpcTests(unittest.TestCase):
    def test_uses_upstream_default_horizon_and_reference_gap(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]])
        )

        self.assertEqual(controller.N, 15)
        self.assertEqual(controller.ref_gap, 3)
        self.assertEqual(controller.ref_traj_len, 6)
        self.assertEqual(controller.opt_controls.shape, (15, 2))
        self.assertEqual(controller.opt_states.shape, (16, 3))
        self.assertFalse(hasattr(controller, "blind_steps"))
        self.assertFalse(hasattr(controller, "prediction_steps"))

    def test_state_contains_pose_only(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            N=3,
            ref_gap=1,
        )

        self.assertEqual(controller.opt_x0.shape, (3, 1))
        self.assertEqual(controller.opt_states.shape, (4, 3))

    def test_reference_sampling_matches_upstream_metric_gap(self):
        path = np.column_stack(
            (np.linspace(0.0, 2.0, 201), np.zeros(201))
        )
        controller = Mpc_controller(
            path,
            N=6,
            desired_v=0.2,
            ref_gap=3,
        )

        refs = controller.find_reference_traj(
            np.zeros(3),
            controller.ref_traj,
        )

        self.assertEqual(len(refs), 3)
        np.testing.assert_allclose(
            np.diff(refs[:, 0]),
            0.06,
            atol=0.01,
        )

    def test_source_matches_upstream_cost_and_zero_yaw_reference(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "controllers.py"
        ).read_text(encoding="utf-8")

        self.assertIn("Q = np.diag([10.0, 10.0, 0.0])", source)
        self.assertIn("R = np.diag([0.02, 0.15])", source)
        self.assertIn("np.zeros((ref_traj.shape[0], 1))", source)
        self.assertNotIn("reference_poses_from_xy", source)
        self.assertNotIn("blind_steps", source)
        self.assertNotIn("update_ref_traj", source)


if __name__ == "__main__":
    unittest.main()
