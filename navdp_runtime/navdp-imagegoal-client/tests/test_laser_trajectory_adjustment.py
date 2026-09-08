import unittest

import numpy as np

from utils_tasks.laser_trajectory_adjustment import adjust_dense_trajectory


class LaserTrajectoryAdjustmentTest(unittest.TestCase):
    def setUp(self):
        self.path = np.column_stack(
            (np.linspace(0.0, 3.0, 301), np.zeros(301))
        )

    def test_safe_path_is_returned_unchanged(self):
        result = adjust_dense_trajectory(
            self.path,
            np.array([[1.0, 2.0]], dtype=np.float64),
        )

        self.assertTrue(result.safe)
        self.assertFalse(result.adjusted)
        self.assertEqual(result.side, "none")
        np.testing.assert_array_equal(result.adjusted_xy, self.path)

    def test_obstacle_on_right_selects_left_and_clears_it(self):
        result = adjust_dense_trajectory(
            self.path,
            np.array([[1.0, -0.10]], dtype=np.float64),
        )

        self.assertTrue(result.safe)
        self.assertTrue(result.adjusted)
        self.assertEqual(result.side, "left")
        self.assertGreater(result.max_offset_m, 0.0)
        self.assertGreaterEqual(result.min_clearance_after_m, 0.32)
        self.assertTrue(np.all(result.offsets_m >= -1e-12))

    def test_offset_ramps_once_and_holds_after_obstacle(self):
        result = adjust_dense_trajectory(
            self.path,
            np.array([[1.0, -0.10]], dtype=np.float64),
        )

        self.assertAlmostEqual(result.offsets_m[0], 0.0)
        self.assertTrue(np.all(np.diff(result.offsets_m) >= -1e-12))
        collision_last = np.flatnonzero(result.collision_mask)[-1]
        np.testing.assert_allclose(
            result.offsets_m[collision_last:],
            result.offsets_m[-1],
            atol=1e-12,
        )

    def test_blocked_both_sides_reports_unsafe_without_inventing_path(self):
        wall_y = np.arange(-0.80, 0.81, 0.02)
        wall = np.column_stack((np.ones_like(wall_y), wall_y))

        result = adjust_dense_trajectory(self.path, wall)

        self.assertFalse(result.safe)
        self.assertFalse(result.adjusted)
        self.assertEqual(result.side, "none")
        np.testing.assert_array_equal(result.adjusted_xy, self.path)

    def test_repeated_dense_waypoints_do_not_break_normal_estimation(self):
        path = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [0.5, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.5, 0.0],
            ]
        )
        obstacles = np.array([[0.75, -0.10]])

        result = adjust_dense_trajectory(path, obstacles)

        self.assertTrue(result.safe)
        self.assertEqual(result.side, "left")
        self.assertGreaterEqual(result.min_clearance_after_m, 0.32)


if __name__ == "__main__":
    unittest.main()
