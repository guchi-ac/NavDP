import unittest
from pathlib import Path

import numpy as np

from scripts.realworld.controllers import (
    Mpc_controller,
    reference_poses_from_xy,
)


class UpstreamNavdpMpcTests(unittest.TestCase):
    def test_reference_pose_yaw_follows_guide_tangent(self):
        poses = reference_poses_from_xy(
            np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
            current_yaw=0.0,
        )
        np.testing.assert_allclose(poses[:, 2], np.pi / 2.0)

    def test_reference_pose_yaw_uses_nearest_equivalent_at_wrap(self):
        angle = np.deg2rad(-179.0)
        points = np.array(
            [[0.0, 0.0], [np.cos(angle), np.sin(angle)]]
        )
        poses = reference_poses_from_xy(
            points,
            current_yaw=np.deg2rad(179.0),
        )
        np.testing.assert_allclose(
            poses[:, 2],
            np.deg2rad(181.0),
            atol=1e-12,
        )

    def test_reference_pose_yaw_handles_repeated_points(self):
        poses = reference_poses_from_xy(
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
            current_yaw=0.3,
        )
        np.testing.assert_allclose(poses[:, 2], 0.0)

    def test_reference_pose_yaw_reuses_nearest_valid_interior_tangent(self):
        poses = reference_poses_from_xy(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            current_yaw=0.0,
        )
        np.testing.assert_allclose(
            poses[:, 2],
            [0.0, 0.0, np.pi / 2.0, np.pi / 2.0, np.pi / 2.0],
            atol=1e-12,
        )

    def test_uses_configured_live_default_horizon_and_reference_gap(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]])
        )

        self.assertEqual(controller.N, 10)
        self.assertEqual(controller.ref_gap, 3)
        self.assertEqual(controller.ref_traj_len, 4)
        self.assertEqual(controller.opt_controls.shape, (10, 2))
        self.assertEqual(controller.opt_states.shape, (11, 3))
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

    def test_update_reference_trajectory_preserves_warm_start(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            N=3,
            ref_gap=1,
        )
        previous_controls = np.full((3, 2), 0.25)
        previous_states = np.full((4, 3), 0.5)
        controller.last_opt_u_controls = previous_controls
        controller.last_opt_x_states = previous_states

        controller.update_ref_traj(
            np.array([[0.0, 0.0], [0.0, 2.0]])
        )

        np.testing.assert_allclose(controller.ref_traj[0], [0.0, 0.0])
        np.testing.assert_allclose(controller.ref_traj[-1], [0.0, 2.0])
        self.assertIs(controller.last_opt_u_controls, previous_controls)
        self.assertIs(controller.last_opt_x_states, previous_states)

    def test_update_dense_reference_uses_adjusted_points_without_resampling(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            N=3,
            ref_gap=1,
        )
        dense = np.array(
            [[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [0.8, 0.2]]
        )
        previous_controls = np.full((3, 2), 0.25)
        previous_states = np.full((4, 3), 0.5)
        controller.last_opt_u_controls = previous_controls
        controller.last_opt_x_states = previous_states

        controller.update_dense_ref_traj(dense)

        np.testing.assert_array_equal(controller.ref_traj, dense)
        self.assertIsNot(controller.ref_traj, dense)
        self.assertIs(controller.last_opt_u_controls, previous_controls)
        self.assertIs(controller.last_opt_x_states, previous_states)

    def test_constructor_accepts_an_already_dense_reference(self):
        dense = np.array(
            [[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [0.8, 0.2]]
        )

        controller = Mpc_controller(
            dense,
            N=3,
            ref_gap=1,
            ref_traj_is_dense=True,
        )

        np.testing.assert_array_equal(controller.ref_traj, dense)
        self.assertIsNot(controller.ref_traj, dense)

    def test_update_dense_reference_rejects_invalid_trajectory(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            N=3,
            ref_gap=1,
        )

        for dense in (
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [np.nan, 0.0]]),
            np.array([0.0, 1.0]),
        ):
            with self.subTest(dense=dense):
                with self.assertRaises(ValueError):
                    controller.update_dense_ref_traj(dense)

    def test_straight_guide_keeps_angular_velocity_small(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            desired_v=0.15,
            v_max=0.15,
            w_max=0.5,
        )
        controls, _ = controller.solve(np.array([0.0, 0.0, 0.0]))
        self.assertGreater(controls[0, 0], 0.05)
        self.assertLess(abs(controls[0, 1]), 0.02)

    def test_right_angle_guide_trades_linear_speed_for_angular_speed(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [0.0, 1.0]]),
            desired_v=0.15,
            v_max=0.15,
            w_max=0.5,
        )
        controls, _ = controller.solve(np.array([0.0, 0.0, 0.0]))
        self.assertLess(controls[0, 0], 0.03)
        self.assertGreater(controls[0, 1], 0.2)

    def test_wrap_boundary_solver_turns_shortest_way_in_both_directions(self):
        for current_degrees, tangent_degrees, expected_sign in (
            (179.0, -179.0, 1.0),
            (-179.0, 179.0, -1.0),
        ):
            with self.subTest(
                current_degrees=current_degrees,
                tangent_degrees=tangent_degrees,
            ):
                tangent = np.deg2rad(tangent_degrees)
                controller = Mpc_controller(
                    np.array(
                        [
                            [0.0, 0.0],
                            [np.cos(tangent), np.sin(tangent)],
                        ]
                    ),
                    desired_v=0.15,
                    v_max=0.15,
                    w_max=0.5,
                )
                controls, _ = controller.solve(
                    np.array([0.0, 0.0, np.deg2rad(current_degrees)])
                )
                self.assertGreater(expected_sign * controls[0, 1], 0.05)

    def test_source_uses_guide_yaw_reference_and_tuned_cost(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "realworld"
            / "controllers.py"
        ).read_text(encoding="utf-8")

        self.assertIn("Q = np.diag([10.0, 10.0, 5.0])", source)
        self.assertIn("R = np.diag([0.02, 0.15])", source)
        self.assertIn(
            "reference_poses_from_xy(ref_traj, x0[2])",
            source,
        )
        self.assertNotIn("np.zeros((ref_traj.shape[0], 1))", source)
        self.assertNotIn("blind_steps", source)


if __name__ == "__main__":
    unittest.main()
