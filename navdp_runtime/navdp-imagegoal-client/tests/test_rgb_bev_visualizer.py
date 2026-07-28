import math
import unittest

import numpy as np

from utils_tasks import rgb_bev_visualizer
from utils_tasks.rgb_bev_visualizer import (
    BevConfig,
    backproject_rgbd_to_base,
    bev_freshness,
    render_mpc_rgb_bev,
    world_xy_to_current_base,
)


class ProjectionTests(unittest.TestCase):
    def test_backprojects_pixel_through_full_rigid_transform(self):
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        rgb[1, 2] = (7, 11, 19)
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 2] = 2.0
        intrinsic = np.array(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
        )
        base_from_camera = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0, 2.0],
                [0.0, -1.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        points, colors, ranges = backproject_rgbd_to_base(
            rgb,
            depth,
            intrinsic,
            base_from_camera,
            BevConfig(sample_stride=1),
        )

        np.testing.assert_allclose(points, [[3.0, 1.0, 0.5]], atol=1e-6)
        np.testing.assert_array_equal(colors, [[7, 11, 19]])
        np.testing.assert_allclose(ranges, [2.0])

    def test_rotates_world_history_into_current_robot_frame(self):
        converted = world_xy_to_current_base(
            np.array([[1.0, 1.0], [1.0, 0.0]]),
            np.array([1.0, 1.0, math.pi / 2]),
        )

        np.testing.assert_allclose(
            converted,
            [[0.0, 0.0], [-1.0, 0.0]],
            atol=1e-7,
        )


class RenderingTests(unittest.TestCase):
    def test_formats_desired_and_actual_velocity_as_separate_lines(self):
        self.assertTrue(hasattr(rgb_bev_visualizer, "velocity_overlay_lines"))
        lines = rgb_bev_visualizer.velocity_overlay_lines(
            desired_velocity=np.array([0.1, -0.128]),
            actual_velocity=np.array([0.094, -0.101]),
            solve_ms=24.5,
        )

        self.assertEqual(
            lines,
            (
                "desired: v=0.100 m/s  w=-0.128 rad/s",
                "actual:  v=0.094 m/s  w=-0.101 rad/s",
                "solve=24.5 ms",
            ),
        )

    def test_formats_missing_actual_velocity_as_nan(self):
        self.assertTrue(hasattr(rgb_bev_visualizer, "velocity_overlay_lines"))
        lines = rgb_bev_visualizer.velocity_overlay_lines(
            desired_velocity=np.array([0.1, 0.2]),
            actual_velocity=None,
            solve_ms=None,
        )

        self.assertEqual(
            lines[1],
            "actual:  v=nan m/s  w=nan rad/s",
        )
        self.assertEqual(lines[2], "solve=nan ms")

    def test_reports_waiting_before_first_odom_and_stale_mpc(self):
        odom_status, mpc_fresh, mpc_status = bev_freshness(
            now=10.0,
            current_odom=None,
            last_odom_time=None,
            mpc_updated_at=None,
            odom_timeout=0.5,
            mpc_timeout=0.5,
        )

        self.assertEqual(odom_status, "ODOM WAITING")
        self.assertFalse(mpc_fresh)
        self.assertEqual(mpc_status, "MPC STALE")

    def test_reports_stale_odom_and_hides_old_mpc(self):
        odom_status, mpc_fresh, mpc_status = bev_freshness(
            now=10.0,
            current_odom=np.zeros(3),
            last_odom_time=9.4,
            mpc_updated_at=9.49,
            odom_timeout=0.5,
            mpc_timeout=0.5,
        )

        self.assertEqual(odom_status, "ODOM STALE")
        self.assertFalse(mpc_fresh)
        self.assertEqual(mpc_status, "MPC STALE")

    def test_reports_fresh_odom_and_mpc_at_timeout_boundary(self):
        odom_status, mpc_fresh, mpc_status = bev_freshness(
            now=10.0,
            current_odom=np.zeros(3),
            last_odom_time=9.5,
            mpc_updated_at=9.5,
            odom_timeout=0.5,
            mpc_timeout=0.5,
        )

        self.assertEqual(odom_status, "ODOM OK")
        self.assertTrue(mpc_fresh)
        self.assertEqual(mpc_status, "MPC OK")

    def test_camera_nearest_sample_wins_same_bev_cell(self):
        rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        depth = np.array([[2.0, 1.0]], dtype=np.float32)
        intrinsic = np.array(
            [[10000.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        base_from_camera = np.eye(4)
        base_from_camera[0, 3] = 2.0
        frame = render_mpc_rgb_bev(
            rgb,
            depth,
            intrinsic,
            base_from_camera,
            current_odom_xy_yaw=None,
            odom_history=np.empty((0, 3)),
            predicted_states=None,
            command=np.zeros(2),
            solve_ms=None,
            odom_status="ODOM WAITING",
            mpc_status="MPC STALE",
            config=BevConfig(
                sample_stride=1,
                min_height_m=0.0,
                max_height_m=3.0,
                splat_radius_px=0,
            ),
        )

        np.testing.assert_array_equal(frame[360, 360], [0, 0, 255])

    def test_draws_cyan_selected_diffusion_under_yellow_guide_points(self):
        frame = render_mpc_rgb_bev(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.eye(3),
            np.eye(4),
            current_odom_xy_yaw=np.zeros(3),
            odom_history=np.empty((0, 3)),
            predicted_states=None,
            selected_diffusion=np.array(
                [[0.50, -0.50], [0.50, 0.50]],
            ),
            active_traj=np.array(
                [[0.50, 0.50], [0.55, 0.50]],
            ),
            command=np.zeros(2),
            solve_ms=None,
            odom_status="ODOM OK",
            mpc_status="MPC OK",
            config=BevConfig(sample_stride=1),
        )

        np.testing.assert_array_equal(frame[495, 405], [255, 255, 0])
        np.testing.assert_array_equal(frame[495, 315], [0, 255, 255])
        np.testing.assert_array_equal(frame[96, 205], [255, 255, 0])
        np.testing.assert_array_equal(frame[96, 325], [0, 255, 255])

    def test_draws_discrete_guide_points_and_robot_over_the_chassis_anchor(self):
        frame = render_mpc_rgb_bev(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.eye(3),
            np.eye(4),
            current_odom_xy_yaw=np.zeros(3),
            odom_history=np.array(
                [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            predicted_states=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ),
            active_traj=np.array(
                [[0.50, 0.50], [0.55, 0.50], [0.60, 0.50], [0.65, 0.50]]
            ),
            command=np.array([0.07, 0.2]),
            solve_ms=12.5,
            odom_status="ODOM OK",
            mpc_status="MPC OK",
            config=BevConfig(sample_stride=1),
        )

        self.assertGreater(frame[585, 360, 1], 200)
        self.assertLess(frame[585, 360, 2], 80)
        self.assertGreater(frame[495, 360, 2], 200)
        self.assertLess(frame[495, 360, 1], 80)
        # The first 0.05 m sample is enlarged, while later samples stay
        # separated at the default 90 px/m BEV scale.
        self.assertGreater(frame[495, 319, 1], 200)
        self.assertGreater(frame[495, 319, 2], 200)
        self.assertGreater(frame[486, 315, 1], 200)
        self.assertGreater(frame[486, 315, 2], 200)
        self.assertTrue((frame[484, 315] < 80).all())
        self.assertGreater(frame[96, 325, 1], 200)
        self.assertGreater(frame[96, 325, 2], 200)

        chassis_anchor = render_mpc_rgb_bev(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.eye(3),
            np.eye(4),
            current_odom_xy_yaw=np.zeros(3),
            odom_history=np.empty((0, 3)),
            predicted_states=None,
            active_traj=np.array([[0.0, 0.0], [0.05, 0.0]]),
            command=np.zeros(2),
            solve_ms=None,
            odom_status="ODOM OK",
            mpc_status="MPC OK",
            config=BevConfig(sample_stride=1),
        )
        self.assertTrue((chassis_anchor[540, 360] > 200).all())
        robot_region = frame[526:555, 346:375]
        self.assertTrue((robot_region > 200).all(axis=2).any())

    def test_status_text_handles_waiting_and_stale_without_paths(self):
        frame = render_mpc_rgb_bev(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.float32),
            np.eye(3),
            np.eye(4),
            current_odom_xy_yaw=None,
            odom_history=np.empty((0, 3)),
            predicted_states=None,
            command=np.zeros(2),
            solve_ms=None,
            odom_status="ODOM WAITING",
            mpc_status="MPC STALE",
            config=BevConfig(sample_stride=1),
        )

        self.assertEqual(frame.shape, (720, 720, 3))


if __name__ == "__main__":
    unittest.main()
