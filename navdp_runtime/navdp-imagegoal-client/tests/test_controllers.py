import math
import unittest

import numpy as np

from scripts.realworld.controllers import Mpc_controller, reference_poses_from_xy


class ReferencePoseTests(unittest.TestCase):
    def test_uses_path_tangent_and_keeps_terminal_heading(self):
        reference_xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        poses = reference_poses_from_xy(reference_xy, current_yaw=0.0)

        np.testing.assert_allclose(poses[:, :2], reference_xy)
        np.testing.assert_allclose(
            poses[:, 2],
            [0.0, math.pi / 2, math.pi / 2, math.pi / 2],
        )

    def test_unwraps_reference_heading_near_current_yaw(self):
        heading = math.radians(-179.0)
        reference_xy = np.array(
            [[0.0, 0.0], [math.cos(heading), math.sin(heading)]]
        )

        poses = reference_poses_from_xy(
            reference_xy,
            current_yaw=math.radians(179.0),
        )

        self.assertAlmostEqual(poses[0, 2], math.radians(181.0))
        self.assertAlmostEqual(poses[1, 2], math.radians(181.0))


class KinematicMpcTests(unittest.TestCase):
    def test_state_contains_pose_only(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            N=3,
            ref_gap=1,
        )

        self.assertEqual(controller.opt_x0.shape, (3, 1))
        self.assertEqual(controller.opt_states.shape, (4, 3))

    def test_blind_steps_extend_diffusion_prediction_horizon(self):
        controller = Mpc_controller(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            N=3,
            blind_steps=2,
            ref_gap=1,
        )

        self.assertEqual(controller.N, 3)
        self.assertEqual(controller.blind_steps, 2)
        self.assertEqual(controller.prediction_steps, 5)
        self.assertEqual(controller.opt_controls.shape, (5, 2))
        self.assertEqual(controller.opt_states.shape, (6, 3))

    def test_ref_gap_distance_sampling_is_unchanged_for_extended_horizon(self):
        path = np.column_stack(
            (np.linspace(0.0, 2.0, 201), np.zeros(201))
        )
        controller = Mpc_controller(
            path,
            N=4,
            blind_steps=2,
            desired_v=0.2,
            ref_gap=2,
        )

        refs = controller.find_reference_traj(
            np.zeros(3),
            controller.ref_traj,
        )

        self.assertEqual(len(refs), 4)
        np.testing.assert_allclose(
            np.diff(refs[:, 0])[:2],
            0.04,
            atol=0.01,
        )

    def test_horizon_parameters_require_integral_values_in_range(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0]])
        invalid_cases = (
            {"N": 0},
            {"N": -1},
            {"N": 2.5},
            {"N": True},
            {"blind_steps": -1},
            {"blind_steps": 1.5},
            {"blind_steps": True},
            {"ref_gap": 0},
            {"ref_gap": -1},
            {"ref_gap": 1.5},
            {"ref_gap": True},
        )

        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    Mpc_controller(path, **changes)

    def test_reference_can_update_only_when_total_dimension_is_unchanged(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0]])
        controller = Mpc_controller(
            path,
            N=3,
            blind_steps=2,
            ref_gap=1,
        )

        controller.update_ref_traj(path, N=4, blind_steps=1)

        self.assertEqual(controller.N, 4)
        self.assertEqual(controller.blind_steps, 1)
        self.assertEqual(controller.prediction_steps, 5)
        with self.assertRaises(ValueError):
            controller.update_ref_traj(path, N=4, blind_steps=2)


if __name__ == "__main__":
    unittest.main()
