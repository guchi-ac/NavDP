import math
import unittest

import numpy as np

from utils_tasks.laser_obstacle_map import make_laser_scan_snapshot
from utils_tasks.live_laser_trajectory import prepare_live_laser_trajectory


class LiveLaserTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.path = np.column_stack(
            (np.linspace(0.0, 3.0, 301), np.zeros(301))
        )
        self.plan_odom = np.zeros(3, dtype=np.float64)

    @staticmethod
    def scan(
        *,
        ranges,
        angle_min,
        angle_increment=0.01,
        received_at=1.0,
        odom=(0.0, 0.0, 0.0),
    ):
        return make_laser_scan_snapshot(
            sequence=7,
            stamp_ns=1_000_000_000,
            received_at=received_at,
            frame_id="laser_frame",
            angle_min=angle_min,
            angle_increment=angle_increment,
            range_min=0.1,
            range_max=16.0,
            ranges=np.asarray(ranges, dtype=np.float32),
            odom_xy_yaw=(
                None if odom is None else np.asarray(odom, dtype=np.float64)
            ),
        )

    def right_obstacle_scan(self, **overrides):
        desired_laser_xy = np.array([1.0 - 0.042, -0.10])
        raw_laser_xy = -desired_laser_xy
        values = {
            "ranges": [np.linalg.norm(raw_laser_xy)],
            "angle_min": math.atan2(raw_laser_xy[1], raw_laser_xy[0]),
        }
        values.update(overrides)
        return self.scan(**values)

    def blocked_wall_scan(self):
        base_angles = np.arange(-0.70, 0.701, 0.02)
        return self.scan(
            ranges=1.0 / np.cos(base_angles),
            angle_min=-math.pi + base_angles[0],
            angle_increment=0.02,
        )

    def prepare(self, scan, *, now=1.10):
        return prepare_live_laser_trajectory(
            dense_xy=self.path,
            scan=scan,
            plan_odom_xy_yaw=self.plan_odom,
            now_monotonic=now,
            timeout_s=0.25,
        )

    def test_missing_scan_stops(self):
        result = self.prepare(None)

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "scan_missing")
        self.assertIsNone(result.trajectory_local_xy)

    def test_stale_scan_stops(self):
        result = self.prepare(
            self.right_obstacle_scan(received_at=1.0),
            now=1.30,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "scan_stale")

    def test_scan_without_odometry_stops(self):
        result = self.prepare(self.right_obstacle_scan(odom=None))

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "scan_odom_missing")

    def test_scan_without_valid_returns_stops(self):
        result = self.prepare(
            self.scan(ranges=[0.0], angle_min=-math.pi)
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "scan_empty")

    def test_right_obstacle_returns_safe_left_adjustment(self):
        result = self.prepare(self.right_obstacle_scan())

        self.assertTrue(result.safe)
        self.assertEqual(result.reason, "ok")
        self.assertEqual(result.scan_sequence, 7)
        self.assertAlmostEqual(result.scan_age_s, 0.10)
        self.assertEqual(result.adjustment.side, "left")
        self.assertGreaterEqual(
            result.adjustment.min_clearance_after_m,
            0.32,
        )
        np.testing.assert_array_equal(
            result.trajectory_local_xy,
            result.adjustment.adjusted_xy,
        )

    def test_blocked_both_sides_stops(self):
        result = self.prepare(self.blocked_wall_scan())

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "adjustment_unsafe")
        self.assertIsNone(result.trajectory_local_xy)


if __name__ == "__main__":
    unittest.main()
