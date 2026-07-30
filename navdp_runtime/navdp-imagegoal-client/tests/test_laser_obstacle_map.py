import math
import unittest

import numpy as np

from utils_tasks.laser_obstacle_map import (
    LaserMapConfig,
    build_current_obstacle_map,
    laser_bev_obstacles,
    laser_scan_record,
    laser_scan_association,
    laser_points_in_target_base,
    make_laser_scan_snapshot,
    nearest_scan_snapshot,
)


def snapshot(
    *,
    sequence=1,
    stamp_ns=1_000_000_000,
    ranges=(1.0,),
    angle_min=0.0,
    angle_increment=0.1,
    range_min=0.1,
    range_max=16.0,
    odom=(0.0, 0.0, 0.0),
):
    return make_laser_scan_snapshot(
        sequence=sequence,
        stamp_ns=stamp_ns,
        received_at=10.0,
        frame_id="laser_frame",
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        ranges=np.asarray(ranges, dtype=np.float32),
        odom_xy_yaw=np.asarray(odom, dtype=np.float64),
    )


class LaserSnapshotTests(unittest.TestCase):
    def test_snapshot_owns_readonly_ranges_and_odom(self):
        ranges = np.array([1.0, 2.0], dtype=np.float32)
        odom = np.array([3.0, 4.0, 0.5])

        result = make_laser_scan_snapshot(
            sequence=7,
            stamp_ns=123,
            received_at=9.0,
            frame_id="laser_frame",
            angle_min=-1.0,
            angle_increment=0.1,
            range_min=0.1,
            range_max=16.0,
            ranges=ranges,
            odom_xy_yaw=odom,
        )
        ranges[:] = 99.0
        odom[:] = 99.0

        np.testing.assert_allclose(result.ranges, [1.0, 2.0])
        np.testing.assert_allclose(result.odom_xy_yaw, [3.0, 4.0, 0.5])
        self.assertFalse(result.ranges.flags.writeable)
        self.assertFalse(result.odom_xy_yaw.flags.writeable)

    def test_rejects_invalid_metadata(self):
        with self.assertRaisesRegex(ValueError, "angle_increment"):
            snapshot(angle_increment=0.0)
        with self.assertRaisesRegex(ValueError, "range bounds"):
            snapshot(range_min=2.0, range_max=1.0)
        with self.assertRaisesRegex(ValueError, "ranges"):
            snapshot(ranges=[])

    def test_pairs_nearest_scan_with_inclusive_slop(self):
        scans = [
            snapshot(sequence=1, stamp_ns=900_000_000),
            snapshot(sequence=2, stamp_ns=1_050_000_000),
        ]

        paired, delta = nearest_scan_snapshot(
            scans,
            target_stamp_ns=1_000_000_000,
            max_delta_s=0.05,
        )

        self.assertEqual(paired.sequence, 2)
        self.assertAlmostEqual(delta, 0.05)

    def test_returns_none_outside_slop(self):
        paired, delta = nearest_scan_snapshot(
            [snapshot(stamp_ns=800_000_000)],
            target_stamp_ns=1_000_000_000,
            max_delta_s=0.10,
        )

        self.assertIsNone(paired)
        self.assertIsNone(delta)

    def test_default_sync_window_accepts_measured_sensor_stamp_offset(self):
        paired, delta = nearest_scan_snapshot(
            [snapshot(sequence=9, stamp_ns=780_000_000)],
            target_stamp_ns=1_000_000_000,
        )

        self.assertEqual(paired.sequence, 9)
        self.assertAlmostEqual(delta, -0.22)

    def test_builds_raw_scan_record_without_losing_pairing_identity(self):
        result = laser_scan_record(
            snapshot(
                sequence=17,
                stamp_ns=2_500_000_000,
                ranges=[0.0, math.nan, 1.25],
                odom=(3.0, 4.0, 0.5),
            ),
            wall_time=20.0,
        )

        self.assertEqual(result["type"], "scan")
        self.assertEqual(result["scan_sequence"], 17)
        self.assertEqual(result["stamp_ns"], 2_500_000_000)
        self.assertEqual(result["frame_id"], "laser_frame")
        np.testing.assert_allclose(
            result["ranges"][[0, 2]],
            [0.0, 1.25],
        )
        self.assertTrue(math.isnan(result["ranges"][1]))
        np.testing.assert_allclose(result["odom"], [3.0, 4.0, 0.5])

    def test_builds_association_fields_and_age(self):
        scan = snapshot(sequence=4, stamp_ns=1_050_000_000)

        result = laser_scan_association(
            scan,
            scan_rgb_dt_s=0.05,
            now_monotonic=10.08,
        )

        self.assertEqual(result["scan_sequence"], 4)
        self.assertEqual(result["scan_stamp_ns"], 1_050_000_000)
        self.assertAlmostEqual(result["scan_rgb_dt_s"], 0.05)
        self.assertAlmostEqual(result["scan_age_s"], 0.08)

    def test_missing_scan_has_explicit_null_association(self):
        result = laser_scan_association(
            None,
            scan_rgb_dt_s=None,
            now_monotonic=10.0,
        )

        self.assertEqual(
            result,
            {
                "scan_sequence": None,
                "scan_stamp_ns": None,
                "scan_rgb_dt_s": None,
                "scan_age_s": None,
            },
        )


class LaserMapTests(unittest.TestCase):
    def test_prepares_only_fresh_scan_hits_for_bev(self):
        scan = snapshot(ranges=[1.0], odom=(0.0, 0.0, 0.0))

        points, status, age = laser_bev_obstacles(
            scan,
            target_odom_xy_yaw=np.array([0.1, 0.0, 0.0]),
            now_monotonic=10.03,
            timeout_s=0.25,
            config=LaserMapConfig(),
        )

        np.testing.assert_allclose(points, [[0.942, 0.0]], atol=1e-6)
        self.assertEqual(status, "LASER OK points=1")
        self.assertAlmostEqual(age, 0.03)

    def test_hides_missing_and_stale_scan_hits_from_bev(self):
        waiting = laser_bev_obstacles(
            None,
            target_odom_xy_yaw=np.zeros(3),
            now_monotonic=10.0,
            timeout_s=0.25,
            config=LaserMapConfig(),
        )
        stale = laser_bev_obstacles(
            snapshot(ranges=[1.0]),
            target_odom_xy_yaw=np.zeros(3),
            now_monotonic=10.26,
            timeout_s=0.25,
            config=LaserMapConfig(),
        )

        self.assertEqual(waiting, (None, "LASER WAITING", None))
        self.assertIsNone(stale[0])
        self.assertEqual(stale[1], "LASER STALE")
        self.assertAlmostEqual(stale[2], 0.26)

    def test_projects_hit_with_mira3_laser_offset(self):
        result = build_current_obstacle_map(
            snapshot(ranges=[1.0]),
            LaserMapConfig(),
        )

        np.testing.assert_allclose(result.obstacle_xy, [[1.042, 0.0]], atol=1e-6)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(np.count_nonzero(result.grid == 100), 1)

    def test_ignores_zero_nonfinite_and_out_of_range_values(self):
        result = build_current_obstacle_map(
            snapshot(
                ranges=[0.0, math.nan, math.inf, 0.05, 17.0, 1.0],
                angle_increment=0.2,
            ),
            LaserMapConfig(),
        )

        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.invalid_count, 5)

    def test_removes_endpoints_inside_robot_self_rectangle(self):
        result = build_current_obstacle_map(
            snapshot(ranges=[0.15, 1.0], angle_increment=math.pi),
            LaserMapConfig(),
        )

        self.assertEqual(result.self_filtered_count, 1)
        self.assertEqual(result.valid_count, 1)
        np.testing.assert_allclose(result.obstacle_xy, [[-0.958, 0.0]], atol=1e-6)

    def test_places_hit_in_expected_grid_cell(self):
        config = LaserMapConfig(resolution_m=0.05)

        result = build_current_obstacle_map(
            snapshot(ranges=[1.0]),
            config,
        )

        row = round((config.forward_m - 1.042) / config.resolution_m)
        col = round(config.lateral_m / config.resolution_m)
        self.assertEqual(result.grid[row, col], 100)

    def test_compensates_scan_pose_into_target_robot_frame(self):
        converted = laser_points_in_target_base(
            np.array([[1.0, 0.0]]),
            source_odom_xy_yaw=np.array([1.0, 0.0, math.pi / 2]),
            target_odom_xy_yaw=np.array([1.0, 0.0, 0.0]),
        )

        np.testing.assert_allclose(converted, [[0.0, 1.0]], atol=1e-7)
