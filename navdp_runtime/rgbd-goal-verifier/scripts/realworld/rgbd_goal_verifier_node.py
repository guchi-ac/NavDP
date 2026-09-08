#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_tasks.rgbd_goal_verifier import RgbdGoalVerifier, VerifierConfig


class RgbdGoalVerifierNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("rgbd_goal_verifier")
        goal_bgr = cv2.imread(args.goal_image, cv2.IMREAD_COLOR)
        if goal_bgr is None:
            raise FileNotFoundError(f"cannot read goal image: {args.goal_image}")

        self.verifier = RgbdGoalVerifier(
            goal_bgr,
            VerifierConfig(
                arrival_distance_m=args.arrival_distance,
                distance_scale=args.distance_scale,
                ratio_test=args.ratio_test,
                min_matches=args.min_matches,
                min_inliers=args.min_inliers,
                min_depth_m=args.min_depth,
                max_depth_m=args.max_depth,
                pnp_reprojection_error_px=args.reprojection_error,
                required_consecutive=args.required_consecutive,
            ),
        )
        self.bridge = CvBridge()
        self.intrinsic = None
        self.period = args.period
        self.last_update = 0.0

        self.info_sub = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            args.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            args.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=1,
            slop=args.sync_slop,
        )
        self.sync.registerCallback(self._rgbd_callback)
        self.get_logger().info(
            "dry-run verifier ready: goal=%s radius=%.2fm distance_scale=%.3f required=%d"
            % (
                args.goal_image,
                args.arrival_distance,
                args.distance_scale,
                args.required_consecutive,
            )
        )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.intrinsic = np.asarray(message.k, dtype=np.float64).reshape(3, 3)

    def _rgbd_callback(self, rgb_message: Image, depth_message: Image) -> None:
        now = time.monotonic()
        if now - self.last_update < self.period:
            return
        self.last_update = now
        if self.intrinsic is None:
            self.get_logger().warning("waiting for CameraInfo")
            return

        try:
            current_bgr = self.bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8")
            depth = np.asarray(
                self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
            )
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            if depth_message.encoding in ("16UC1", "mono16"):
                depth_m = depth.astype(np.float32) / 1000.0
            elif depth_message.encoding == "32FC1":
                depth_m = depth.astype(np.float32)
            else:
                raise ValueError(f"unsupported depth encoding: {depth_message.encoding}")

            result = self.verifier.update(current_bgr, depth_m, self.intrinsic)
            distance = "nan" if result.distance_m is None else f"{result.distance_m:.3f}"
            self.get_logger().info(
                "status=%s candidate=%s arrived=%s distance_m=%s "
                "features=%d matches=%d depth_matches=%d inliers=%d streak=%d"
                % (
                    result.reason,
                    result.candidate,
                    result.arrived,
                    distance,
                    result.current_features,
                    result.good_matches,
                    result.depth_matches,
                    result.inliers,
                    result.consecutive,
                )
            )
        except Exception as error:
            self.get_logger().error(f"verification frame failed: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run RGB-D ImageGoal arrival verifier")
    parser.add_argument("--goal-image", required=True)
    parser.add_argument("--rgb-topic", default="/cam_head/d435/color/image_raw")
    parser.add_argument(
        "--depth-topic",
        default="/cam_head/d435/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/cam_head/d435/color/camera_info",
    )
    parser.add_argument("--period", type=float, default=0.3)
    parser.add_argument("--sync-slop", type=float, default=0.1)
    parser.add_argument("--arrival-distance", type=float, default=0.01)
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=10.0,
        help="multiply the raw PnP translation by this calibration factor",
    )
    parser.add_argument("--ratio-test", type=float, default=0.65)
    parser.add_argument("--min-matches", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--reprojection-error", type=float, default=2.0)
    parser.add_argument("--required-consecutive", type=int, default=8)
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def main() -> None:
    args = parse_args()
    rclpy.init(args=[])
    node = None
    try:
        node = RgbdGoalVerifierNode(args)
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
