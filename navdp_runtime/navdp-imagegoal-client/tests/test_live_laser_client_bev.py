import math
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.realworld.navdp_imagegoal_client import (
    FrameSnapshot,
    NavdpImageGoalClient,
)
from utils_tasks.laser_obstacle_map import (
    LaserMapConfig,
    make_laser_scan_snapshot,
)
from utils_tasks.rgb_bev_visualizer import BevConfig


class LiveLaserClientBevTests(unittest.TestCase):
    def test_bev_draws_latest_scan_when_rgb_frame_has_no_paired_scan(self):
        now = time.monotonic()
        desired_base_xy = np.array([1.0, 0.5])
        raw_laser_xy = -(desired_base_xy - np.array([0.042, 0.0]))
        latest_scan = make_laser_scan_snapshot(
            sequence=7,
            stamp_ns=1,
            received_at=now,
            frame_id="laser_frame",
            angle_min=math.atan2(raw_laser_xy[1], raw_laser_xy[0]),
            angle_increment=0.01,
            range_min=0.1,
            range_max=16.0,
            ranges=np.array(
                [np.linalg.norm(raw_laser_xy)],
                dtype=np.float32,
            ),
            odom_xy_yaw=np.zeros(3),
        )
        snapshot = FrameSnapshot(
            sequence=3,
            stamp_ns=2,
            rgb_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_m=np.zeros((2, 2), dtype=np.float32),
            intrinsic=np.eye(3),
            odom_xy_yaw=np.zeros(3),
            camera_xy_yaw=np.zeros(3),
            base_from_camera=np.eye(4),
            received_at=now,
            laser_snapshot=None,
            scan_rgb_dt_s=None,
        )
        client = object.__new__(NavdpImageGoalClient)
        client.data_lock = threading.Lock()
        client.last_odom_time = now
        client.odom_history = []
        client.latest_odom_twist = None
        client.latest_mpc_visualization = None
        client.latest_scan = latest_scan
        client.last_laser_map_error_log = 0.0
        client.laser_map_config = LaserMapConfig()
        client.bev_config = BevConfig(sample_stride=1)
        client.args = SimpleNamespace(
            scan_timeout=0.25,
            odom_timeout=0.5,
            mpc_bev_timeout=0.5,
        )

        frame = client._render_mpc_bev(snapshot)

        magenta = np.all(
            frame[446:455, 311:320] == [255, 0, 255],
            axis=2,
        )
        self.assertTrue(magenta.any())


if __name__ == "__main__":
    unittest.main()
