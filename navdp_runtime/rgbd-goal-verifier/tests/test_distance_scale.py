import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "realworld"))

from rgbd_goal_verifier_node import parse_args
from utils_tasks.rgbd_goal_verifier import VerifierConfig, estimate_relative_camera_pose


class DistanceScaleTests(unittest.TestCase):
    def test_pose_distance_uses_configured_scale(self):
        intrinsic = np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rng = np.random.default_rng(7)
        depths = rng.uniform(1.5, 4.0, 100)
        current_pixels = np.column_stack(
            (rng.uniform(80, 560, 100), rng.uniform(60, 420, 100))
        )
        object_points = np.column_stack(
            (
                (current_pixels[:, 0] - 320.0) * depths / 600.0,
                (current_pixels[:, 1] - 240.0) * depths / 600.0,
                depths,
            )
        )
        goal_points = object_points + np.array([0.06, 0.0, 0.0])
        goal_pixels = np.column_stack(
            (
                600.0 * goal_points[:, 0] / goal_points[:, 2] + 320.0,
                600.0 * goal_points[:, 1] / goal_points[:, 2] + 240.0,
            )
        )
        depth_image = np.zeros((480, 640), dtype=np.float32)
        for (u, v), depth in zip(current_pixels, depths):
            depth_image[int(round(v)), int(round(u))] = depth

        estimate = estimate_relative_camera_pose(
            current_pixels,
            goal_pixels,
            depth_image,
            intrinsic,
            VerifierConfig(
                distance_scale=10.0,
                min_matches=8,
                min_inliers=6,
                pnp_reprojection_error_px=1.0,
            ),
        )

        self.assertTrue(estimate.success)
        self.assertAlmostEqual(estimate.distance_m, 0.60, places=4)

    def test_node_defaults_to_measured_distance_scale(self):
        with patch.object(
            sys,
            "argv",
            ["rgbd_goal_verifier_node.py", "--goal-image", "goal.jpg"],
        ):
            args = parse_args()

        self.assertEqual(args.distance_scale, 10.0)


if __name__ == "__main__":
    unittest.main()
