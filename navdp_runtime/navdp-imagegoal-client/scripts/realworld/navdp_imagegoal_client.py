#!/usr/bin/env python3
import os

# OpenBLAS reads this once when NumPy is imported.
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("NAVDP_BLAS_THREADS", "2")

import argparse
import copy
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, TwistStamped
from interfaces.action import Skill, Torso
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time as RosTime
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformException,
    TransformListener,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from controllers import Mpc_controller
from utils_tasks.client_utils import imagegoal_step, navigator_close, navigator_reset
from utils_tasks.laser_obstacle_map import (
    DEFAULT_SCAN_SYNC_SLOP_S,
    MIRA3_LASER_YAW_RAD,
    LaserMapConfig,
    LaserScanSnapshot,
    laser_bev_obstacles,
    laser_scan_association,
    laser_scan_record,
    make_laser_scan_snapshot,
    nearest_scan_snapshot,
)
from utils_tasks.live_laser_trajectory import (
    live_test_max_v,
    prepare_live_laser_trajectory,
)
from utils_tasks.rgb_bev_visualizer import (
    BevConfig,
    bev_freshness,
    render_mpc_rgb_bev,
)
from utils_tasks.rgbd_goal_verifier import RgbdGoalVerifier, VerifierConfig
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.wheeled_client_core import (
    JsonlWriter,
    PostureActionRunner,
    SelectedDiffusionInstallState,
    camera_pose_from_transform,
    control_stop_reason,
    finalize_mp4,
    normalize_tracking_trajectory,
    put_latest,
    resize_rgbd_for_visualization,
    run_navdp_startup,
    tracking_generation_is_current,
    trajectory_to_world,
    transform_matrix_from_translation_quaternion,
    yaw_from_quaternion,
)


@dataclass
class FrameSnapshot:
    sequence: int
    stamp_ns: int
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    odom_xy_yaw: Optional[np.ndarray]
    camera_xy_yaw: np.ndarray
    base_from_camera: np.ndarray
    received_at: float
    laser_snapshot: Optional[LaserScanSnapshot]
    scan_rgb_dt_s: Optional[float]


@dataclass(frozen=True)
class VisualizationState:
    trajectory: np.ndarray
    trajectory_prefix: np.ndarray
    candidates: np.ndarray
    values: np.ndarray
    critic: float
    arrived: bool


@dataclass(frozen=True)
class VisualizationRequest:
    snapshot: FrameSnapshot
    state: Optional[VisualizationState]


@dataclass(frozen=True)
class MpcVisualizationSnapshot:
    predicted_states: np.ndarray
    active_traj: np.ndarray
    selected_diffusion: Optional[np.ndarray]
    command: np.ndarray
    solve_ms: float
    updated_at: float


class NavdpImageGoalClient(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("navdp_imagegoal_client")
        goal_bgr = cv2.imread(args.goal_image, cv2.IMREAD_COLOR)
        if goal_bgr is None:
            raise FileNotFoundError(f"cannot read goal image: {args.goal_image}")

        self.args = args
        self.control_max_v = live_test_max_v(args.max_v)
        self.goal_bgr = goal_bgr
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.torso_action_client = None
        self.posture_action_runner = None
        self.visualizer = VisualizationManager(history_size=5)
        self.visualization_output = Path(args.visualization_output)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        video_dir = Path(args.visualization_video_dir).expanduser()
        video_name = f"{run_stamp}_navdp_footprint.mp4"
        self.visualization_video_output = video_dir / video_name
        self.visualization_video_temporary = video_dir / (
            self.visualization_video_output.stem + ".partial.mp4"
        )
        self.visualization_video_writer = None
        self.mpc_bev_video_output = video_dir / f"{run_stamp}_mpc_rgb_bev.mp4"
        self.mpc_bev_video_temporary = (
            video_dir / f"{run_stamp}_mpc_rgb_bev.partial.mp4"
        )
        self.mpc_bev_video_writer = None
        self.mpc_bev_video_failed = False
        self.bev_config = BevConfig(sample_stride=args.bev_sample_stride)
        for name in (
            "laser_x",
            "laser_y",
            "laser_yaw",
            "scan_sync_slop",
            "scan_timeout",
            "laser_map_resolution",
        ):
            if not np.isfinite(getattr(args, name)):
                raise ValueError(f"{name} must be finite")
        if args.scan_sync_slop < 0.0:
            raise ValueError("scan_sync_slop must be non-negative")
        if args.scan_timeout <= 0.0:
            raise ValueError("scan_timeout must be positive")
        if args.laser_map_resolution <= 0.0:
            raise ValueError("laser_map_resolution must be positive")
        if not args.laser_frame:
            raise ValueError("laser_frame must be non-empty")
        self.laser_map_config = LaserMapConfig(
            resolution_m=args.laser_map_resolution,
            laser_x_m=args.laser_x,
            laser_y_m=args.laser_y,
            laser_yaw_rad=args.laser_yaw,
        )
        mpc_log_dir = Path(args.mpc_log_dir).expanduser()
        self.mpc_diagnostics = JsonlWriter(
            mpc_log_dir / f"{run_stamp}_mpc.jsonl"
        )
        self.mpc_diagnostics_failed = False
        self.verifier = RgbdGoalVerifier(
            goal_bgr,
            VerifierConfig(
                arrival_distance_m=args.arrival_distance,
                min_matches=args.min_matches,
                min_inliers=args.min_inliers,
                required_consecutive=args.arrival_consecutive,
            ),
        )
        self.data_lock = threading.Lock()
        self.mpc_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.intrinsic = None
        self.latest_odom = None
        self.latest_odom_twist = None
        self.last_odom_time = None
        self.odom_history = deque(maxlen=600)
        self.latest_frame = None
        self.frame_sequence = 0
        self.latest_scan: Optional[LaserScanSnapshot] = None
        self.recent_scans = deque(maxlen=50)
        self.scan_sequence = 0
        self.last_scan_error_log = 0.0
        self.last_laser_map_error_log = 0.0
        self.mpc = None
        self.installed_active_traj: Optional[np.ndarray] = None
        self.selected_diffusion_state = SelectedDiffusionInstallState()
        self.last_plan_time = None
        self.plan_sequence = 0
        self.latest_plan_id = None
        self.arrival_blocked = False
        self.trajectory_ready = False
        self.trajectory_generation = 0
        self.server_initialized = False
        self.last_control_log = 0.0
        self.last_control_reason = None
        self.last_tf_error_log = 0.0
        self.latest_mpc_visualization = None
        self.visualization_state = None
        self.visualization_queue = queue.Queue(maxsize=1)
        self.local_nav_goal_handle = None
        self.local_nav_result_future = None

        self.info_sub = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            args.odom_topic,
            self._odom_callback,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            args.scan_topic,
            self._scan_callback,
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
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=1,
            slop=args.sync_slop,
        )
        self.synchronizer.registerCallback(self._rgbd_callback)
        self.control_pub = self.create_publisher(
            TwistStamped,
            args.cmd_topic,
            10,
        )
        self.visualization_pub = self.create_publisher(
            Image,
            args.visualization_topic,
            2,
        )
        self.local_nav_action_client = ActionClient(
            self,
            Skill,
            "/skill_behavior_tree",
        )

        self.planning_thread = threading.Thread(
            target=self._planning_loop,
            name="navdp_planning",
            daemon=True,
        )
        self.control_thread = threading.Thread(
            target=self._control_loop,
            name="navdp_control",
            daemon=True,
        )
        self.visualization_thread = threading.Thread(
            target=self._visualization_loop,
            name="navdp_visualization",
            daemon=True,
        )

    def start(self) -> None:
        if self.args.enable_control:
            self.torso_action_client = ActionClient(
                self,
                Torso,
                "/Torso/torso_action_service",
            )
            self.posture_action_runner = PostureActionRunner(
                node=self,
                action_client=self.torso_action_client,
                timeout=self.args.posture_timeout,
                spin_until_future_complete=rclpy.spin_until_future_complete,
                goal_factory=Torso.Goal,
                succeeded_status=GoalStatus.STATUS_SUCCEEDED,
            )
        run_navdp_startup(
            enable_control=self.args.enable_control,
            align_posture=self._set_navdp_posture,
            publish_camera_tf=self._publish_d435_mount_transform,
            start_local_nav=self._start_local_nav,
            start_workers=self._start_workers,
        )

        args = self.args
        mode = "ENABLED" if args.enable_control else "DRY-RUN"
        self.get_logger().info(
            "client ready: mode=%s goal=%s cmd=%s camera_tf=%s<-%s "
            "scan=%s laser_frame=%s laser_xy_yaw=(%.3f,%.3f,%.3f) "
            "video=%s mpc_log=%s requested_max_v=%.2f "
            "control_max_v=%.2f max_w=%.2f"
            % (
                mode,
                args.goal_image,
                args.cmd_topic,
                args.base_frame,
                args.camera_frame,
                args.scan_topic,
                args.laser_frame,
                args.laser_x,
                args.laser_y,
                args.laser_yaw,
                self.visualization_video_output,
                self.mpc_diagnostics.path,
                args.max_v,
                self.control_max_v,
                args.max_w,
            )
        )
        other_publishers = max(0, self.count_publishers(args.cmd_topic) - 1)
        if other_publishers:
            self.get_logger().warning(
                f"{args.cmd_topic} has {other_publishers} other publishers; keep other navigation commands idle"
            )

    def _start_workers(self) -> None:
        self.planning_thread.start()
        self.control_thread.start()
        self.visualization_thread.start()

    def _set_navdp_posture(self) -> None:
        if self.posture_action_runner is None:
            raise RuntimeError("posture action runner is unavailable")
        self.posture_action_runner.run()
        self.get_logger().info(
            "NavDP posture aligned: torso_yaw=0.0 deg "
            "head_yaw=0.0 deg head_pitch=-19.7795845 deg"
        )

    def _cancel_posture(self) -> None:
        if self.posture_action_runner is None:
            return
        try:
            self.posture_action_runner.cancel()
        except Exception as error:
            self.get_logger().warning(
                f"failed to cancel posture action goal: {error}"
            )

    def _start_local_nav(self) -> None:
        if not self.args.enable_control:
            return
        if not self.local_nav_action_client.wait_for_server(
            timeout_sec=self.args.local_nav_timeout
        ):
            raise RuntimeError(
                "local_nav action server unavailable: /skill_behavior_tree"
            )

        goal = Skill.Goal()
        goal.head.action_name = "local_nav"
        goal_future = self.local_nav_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self,
            goal_future,
            timeout_sec=self.args.local_nav_timeout,
        )
        if not goal_future.done():
            raise RuntimeError("timed out sending local_nav action goal")
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("local_nav action goal was rejected")

        self.local_nav_goal_handle = goal_handle
        self.local_nav_result_future = goal_handle.get_result_async()
        self.get_logger().info("local_nav action goal accepted")

    def _cancel_local_nav(self) -> None:
        if self.local_nav_goal_handle is None:
            return
        if (
            self.local_nav_result_future is not None
            and self.local_nav_result_future.done()
        ):
            self.local_nav_goal_handle = None
            return

        cancel_future = self.local_nav_goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(
            self,
            cancel_future,
            timeout_sec=self.args.local_nav_timeout,
        )
        if not cancel_future.done():
            self.get_logger().warning("timed out canceling local_nav action goal")
        elif cancel_future.result().goals_canceling:
            self.get_logger().info("local_nav action goal cancellation accepted")
        else:
            self.get_logger().warning("local_nav action goal cancellation rejected")
        self.local_nav_goal_handle = None

    def _camera_info_callback(self, message: CameraInfo) -> None:
        with self.data_lock:
            self.intrinsic = np.asarray(message.k, dtype=np.float64).reshape(3, 3)

    def _odom_callback(self, message: Odometry) -> None:
        orientation = message.pose.pose.orientation
        pose = np.array(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                yaw_from_quaternion(
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ),
            ],
            dtype=np.float64,
        )
        twist = np.array(
            [
                message.twist.twist.linear.x,
                message.twist.twist.angular.z,
            ],
            dtype=np.float64,
        )
        with self.data_lock:
            self.latest_odom = pose
            self.latest_odom_twist = twist
            self.last_odom_time = time.monotonic()
            self.odom_history.append(pose.copy())

    def _scan_callback(self, message: LaserScan) -> None:
        received_at = time.monotonic()
        try:
            if message.header.frame_id != self.args.laser_frame:
                raise ValueError(
                    "unexpected laser frame %r, expected %r"
                    % (message.header.frame_id, self.args.laser_frame)
                )
            stamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            with self.data_lock:
                next_sequence = self.scan_sequence + 1
                snapshot = make_laser_scan_snapshot(
                    sequence=next_sequence,
                    stamp_ns=stamp_ns,
                    received_at=received_at,
                    frame_id=message.header.frame_id,
                    angle_min=message.angle_min,
                    angle_increment=message.angle_increment,
                    range_min=message.range_min,
                    range_max=message.range_max,
                    ranges=np.asarray(message.ranges, dtype=np.float32),
                    odom_xy_yaw=(
                        None
                        if self.latest_odom is None
                        else self.latest_odom.copy()
                    ),
                )
                self.scan_sequence = next_sequence
                self.latest_scan = snapshot
                self.recent_scans.append(snapshot)
            self._write_diagnostic(
                laser_scan_record(snapshot, wall_time=time.time())
            )
        except Exception as error:
            if received_at - self.last_scan_error_log >= 2.0:
                self.get_logger().error(f"laser scan rejected: {error}")
                self.last_scan_error_log = received_at

    def _write_diagnostic(self, record: dict) -> None:
        if self.mpc_diagnostics_failed:
            return
        try:
            self.mpc_diagnostics.write(record)
        except Exception as error:
            self.mpc_diagnostics_failed = True
            self.get_logger().error(f"MPC diagnostic logging disabled: {error}")

    def _invalidate_tracking_state(self) -> None:
        with self.data_lock:
            self.trajectory_generation += 1
            self.trajectory_ready = False
            self.latest_mpc_visualization = None
        with self.mpc_lock:
            self.selected_diffusion_state = (
                self.selected_diffusion_state.clear()
            )
            self.mpc = None
            self.installed_active_traj = None

    def _publish_d435_mount_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.args.base_frame
        transform.child_frame_id = "d435_link"
        transform.transform.translation.x = 0.092070325
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 1.254818867
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.171753592
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 0.985139941
        self.static_tf_broadcaster.sendTransform(transform)
        self.get_logger().info(
            "static camera TF published: %s -> d435_link (head pitch down 20 deg)"
            % self.args.base_frame
        )

    def _camera_transform_in_base(self, stamp) -> Tuple[np.ndarray, np.ndarray]:
        transform = self.tf_buffer.lookup_transform(
            self.args.base_frame,
            self.args.camera_frame,
            RosTime.from_msg(stamp),
            timeout=Duration(seconds=self.args.tf_timeout),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        translation_xyz = np.array(
            [translation.x, translation.y, translation.z],
            dtype=np.float64,
        )
        rotation_xyzw = np.array(
            [rotation.x, rotation.y, rotation.z, rotation.w],
            dtype=np.float64,
        )
        base_from_camera = transform_matrix_from_translation_quaternion(
            translation_xyz,
            rotation_xyzw,
        )
        return (
            camera_pose_from_transform(translation_xyz, rotation_xyzw),
            base_from_camera,
        )

    def _rgbd_callback(self, rgb_message: Image, depth_message: Image) -> None:
        try:
            rgb_stamp_ns = (
                int(rgb_message.header.stamp.sec) * 1_000_000_000
                + int(rgb_message.header.stamp.nanosec)
            )
            rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8")
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

            try:
                camera_xy_yaw, base_from_camera = (
                    self._camera_transform_in_base(rgb_message.header.stamp)
                )
            except TransformException as error:
                now = time.monotonic()
                if now - self.last_tf_error_log >= 2.0:
                    self.get_logger().error(
                        "camera TF unavailable from %s: %s"
                        % (self.args.base_frame, error)
                    )
                    self.last_tf_error_log = now
                return

            with self.data_lock:
                if self.intrinsic is None:
                    return
                laser_snapshot, scan_rgb_dt_s = nearest_scan_snapshot(
                    self.recent_scans,
                    rgb_stamp_ns,
                    self.args.scan_sync_slop,
                )
                self.frame_sequence += 1
                snapshot = FrameSnapshot(
                    sequence=self.frame_sequence,
                    stamp_ns=rgb_stamp_ns,
                    rgb_bgr=np.asarray(rgb_bgr).copy(),
                    depth_m=depth_m.copy(),
                    intrinsic=self.intrinsic.copy(),
                    odom_xy_yaw=None
                    if self.latest_odom is None
                    else self.latest_odom.copy(),
                    camera_xy_yaw=camera_xy_yaw,
                    base_from_camera=base_from_camera.copy(),
                    received_at=time.monotonic(),
                    laser_snapshot=laser_snapshot,
                    scan_rgb_dt_s=scan_rgb_dt_s,
                )
                self.latest_frame = snapshot
            self._queue_visualization(snapshot)
        except Exception as error:
            self.get_logger().error(f"RGB-D callback failed: {error}")

    def _planning_loop(self) -> None:
        last_sequence = -1
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            with self.data_lock:
                snapshot = copy.deepcopy(self.latest_frame)
            if snapshot is None or snapshot.sequence == last_sequence:
                self.stop_event.wait(0.02)
                continue
            last_sequence = snapshot.sequence

            active_traj = None
            critic_max = None
            critic_safe = False
            local_xy = None
            raw_selected_world_xy = None
            adjusted_local_xy = None
            laser_result = None
            laser_diagnostic = {
                "laser_adjustment_reason": "not_run",
                "laser_adjustment_scan_sequence": None,
                "laser_adjustment_scan_age_s": None,
                "laser_adjustment_side": None,
                "laser_clearance_before_m": None,
                "laser_clearance_after_m": None,
                "laser_max_offset_m": None,
                "adjusted_local_xy": None,
            }
            trajectory_prefix = np.empty((0, 2), dtype=np.float64)
            candidate_world_xy = np.empty((0, 0, 2), dtype=np.float64)
            candidate_values = np.empty(0, dtype=np.float64)
            planning_error = None
            try:
                if snapshot.odom_xy_yaw is None:
                    raise RuntimeError("no odometry snapshot for planned frame")
                arrival = self.verifier.update(
                    snapshot.rgb_bgr,
                    snapshot.depth_m,
                    snapshot.intrinsic,
                )
                with self.data_lock:
                    self.arrival_blocked = arrival.candidate or arrival.arrived
                self.get_logger().info(
                    "arrival=%s candidate=%s distance=%s inliers=%d streak=%d"
                    % (
                        arrival.arrived,
                        arrival.candidate,
                        "nan"
                        if arrival.distance_m is None
                        else f"{arrival.distance_m:.3f}",
                        arrival.inliers,
                        arrival.consecutive,
                    )
                )
                if arrival.arrived:
                    state = self.visualization_state
                    if state is not None:
                        self.visualization_state = VisualizationState(
                            trajectory=state.trajectory,
                            trajectory_prefix=state.trajectory_prefix,
                            candidates=state.candidates,
                            values=state.values,
                            critic=state.critic,
                            arrived=True,
                        )
                        self._queue_visualization(snapshot)
                    self.stop_event.wait(self.args.plan_period)
                    continue

                if not self.server_initialized:
                    algo = navigator_reset(
                        snapshot.intrinsic,
                        stop_threshold=self.args.critic_threshold,
                        batch_size=1,
                        port=self.args.server_port,
                    )
                    self.server_initialized = True
                    self.get_logger().info(f"NavDP initialized: algo={algo}")

                trajectory, all_trajectories, all_values = imagegoal_step(
                    self.goal_bgr[None, ...],
                    snapshot.rgb_bgr[None, ...],
                    snapshot.depth_m[None, ...],
                    port=self.args.server_port,
                )

                candidate_local = np.asarray(all_trajectories, dtype=np.float64)
                if candidate_local.ndim == 4:
                    candidate_local = candidate_local[0]
                if candidate_local.ndim != 3:
                    raise ValueError(
                        "invalid NavDP candidate shape: %s"
                        % (candidate_local.shape,)
                    )
                candidate_values = np.asarray(all_values, dtype=np.float64)
                if candidate_values.ndim > 1 and candidate_values.shape[0] == 1:
                    candidate_values = candidate_values[0]
                candidate_values = candidate_values.reshape(-1)
                critic_max = float(np.max(all_values))
                critic_safe = critic_max >= self.args.critic_threshold

                raw_local_xy = np.asarray(trajectory, dtype=np.float64)
                if raw_local_xy.ndim == 3:
                    raw_local_xy = raw_local_xy[0]
                raw_local_xy = raw_local_xy[:, :2]
                local_xy = raw_local_xy
                if len(local_xy) < 2 or not np.isfinite(local_xy).all():
                    raise ValueError(f"invalid NavDP trajectory shape: {local_xy.shape}")
                raw_selected_world_xy = trajectory_to_world(
                    raw_local_xy,
                    snapshot.odom_xy_yaw,
                )
                dense_local_xy = Mpc_controller.make_ref_denser(
                    None,
                    raw_local_xy,
                )
                with self.data_lock:
                    latest_scan = self.latest_scan
                laser_result = prepare_live_laser_trajectory(
                    dense_xy=dense_local_xy,
                    scan=latest_scan,
                    plan_odom_xy_yaw=snapshot.odom_xy_yaw,
                    now_monotonic=time.monotonic(),
                    timeout_s=self.args.scan_timeout,
                    laser_config=self.laser_map_config,
                )
                adjustment = laser_result.adjustment
                laser_diagnostic.update(
                    {
                        "laser_adjustment_reason": laser_result.reason,
                        "laser_adjustment_scan_sequence": (
                            laser_result.scan_sequence
                        ),
                        "laser_adjustment_scan_age_s": (
                            laser_result.scan_age_s
                        ),
                        "laser_adjustment_side": (
                            None if adjustment is None else adjustment.side
                        ),
                        "laser_clearance_before_m": (
                            None
                            if adjustment is None
                            else adjustment.min_clearance_before_m
                        ),
                        "laser_clearance_after_m": (
                            None
                            if adjustment is None
                            else adjustment.min_clearance_after_m
                        ),
                        "laser_max_offset_m": (
                            None
                            if adjustment is None
                            else adjustment.max_offset_m
                        ),
                        "adjusted_local_xy": (
                            laser_result.trajectory_local_xy
                        ),
                    }
                )
                candidate_world_xy = np.asarray(
                    [
                        trajectory_to_world(
                            candidate[:, :2],
                            snapshot.odom_xy_yaw,
                            camera_x=snapshot.camera_xy_yaw[0],
                            camera_y=snapshot.camera_xy_yaw[1],
                            camera_yaw=snapshot.camera_xy_yaw[2],
                        )
                        for candidate in candidate_local
                    ]
                )
                if critic_safe and laser_result.safe:
                    adjusted_local_xy = laser_result.trajectory_local_xy
                    active_traj = normalize_tracking_trajectory(
                        trajectory_to_world(
                            adjusted_local_xy,
                            snapshot.odom_xy_yaw,
                        )
                    )
                self.visualization_state = VisualizationState(
                    trajectory=(
                        np.empty((0, 2), dtype=np.float64)
                        if active_traj is None
                        else active_traj.copy()
                    ),
                    trajectory_prefix=trajectory_prefix,
                    candidates=candidate_world_xy.copy(),
                    values=candidate_values.copy(),
                    critic=critic_max,
                    arrived=False,
                )
                self._queue_visualization(snapshot)
                if not critic_safe:
                    self.get_logger().warning(
                        "NavDP candidate below critic threshold: "
                        "max=%.3f threshold=%.3f"
                        % (
                            critic_max,
                            self.args.critic_threshold,
                        )
                    )
                elif not laser_result.safe:
                    self.get_logger().warning(
                        "Laser trajectory adjustment rejected plan: "
                        f"reason={laser_result.reason}"
                    )
            except Exception as error:
                planning_error = error
                self.get_logger().error(f"planning failed: {error}")

            if active_traj is None:
                self._invalidate_tracking_state()
                diagnostic_time = time.monotonic()
                self._write_diagnostic(
                    {
                        "type": "plan",
                        "wall_time": time.time(),
                        "monotonic_time": diagnostic_time,
                        "frame_sequence": snapshot.sequence,
                        "critic": critic_max,
                        "snapshot_odom": snapshot.odom_xy_yaw,
                        "camera_pose": snapshot.camera_xy_yaw,
                        "selected_local_xy": local_xy,
                        "raw_selected_world_xy": raw_selected_world_xy,
                        "active_traj": None,
                        **laser_diagnostic,
                        "planning_error": (
                            None if planning_error is None else str(planning_error)
                        ),
                        **laser_scan_association(
                            snapshot.laser_snapshot,
                            scan_rgb_dt_s=snapshot.scan_rgb_dt_s,
                            now_monotonic=diagnostic_time,
                        ),
                    }
                )
            else:
                try:
                    next_active_traj = np.asarray(active_traj).copy()
                    next_active_traj.setflags(write=False)
                    with self.mpc_lock:
                        if self.mpc is None:
                            next_mpc = Mpc_controller(
                                active_traj,
                                desired_v=self.control_max_v * 0.8,
                                v_max=self.control_max_v,
                                w_max=self.args.max_w,
                                ref_traj_is_dense=True,
                            )
                            self.mpc = next_mpc
                        else:
                            self.mpc.update_dense_ref_traj(active_traj)
                        next_selected_state = (
                            SelectedDiffusionInstallState()
                            .stage(raw_selected_world_xy, True)
                            .commit()
                        )
                        self.installed_active_traj = next_active_traj
                        self.selected_diffusion_state = next_selected_state
                        with self.data_lock:
                            plan_ready = time.monotonic()
                            self.trajectory_generation += 1
                            self.trajectory_ready = True
                            self.last_plan_time = plan_ready
                            self.plan_sequence += 1
                            self.latest_plan_id = self.plan_sequence
                            plan_id = self.latest_plan_id
                except Exception as error:
                    self._invalidate_tracking_state()
                    reason = "active_trajectory_error"
                    self.get_logger().error(
                        f"failed to install active trajectory: {error}"
                    )
                    elapsed = time.monotonic() - cycle_start
                    self.stop_event.wait(
                        max(0.0, self.args.plan_period - elapsed)
                    )
                    continue
                if planning_error is not None and self.visualization_state is not None:
                    state = self.visualization_state
                    self.visualization_state = VisualizationState(
                        trajectory=active_traj.copy(),
                        trajectory_prefix=state.trajectory_prefix,
                        candidates=state.candidates,
                        values=state.values,
                        critic=state.critic,
                        arrived=False,
                    )
                    self._queue_visualization(snapshot)
                self._write_diagnostic(
                    {
                        "type": "plan",
                        "wall_time": time.time(),
                        "monotonic_time": plan_ready,
                        "plan_id": plan_id,
                        "frame_sequence": snapshot.sequence,
                        "critic": critic_max,
                        "planning_ms": (plan_ready - cycle_start) * 1000.0,
                        "snapshot_odom": snapshot.odom_xy_yaw,
                        "camera_pose": snapshot.camera_xy_yaw,
                        "selected_local_xy": local_xy,
                        "raw_selected_world_xy": raw_selected_world_xy,
                        "active_traj": active_traj,
                        **laser_diagnostic,
                        "mpc_horizon": self.mpc.N,
                        "planning_error": (
                            None if planning_error is None else str(planning_error)
                        ),
                        **laser_scan_association(
                            snapshot.laser_snapshot,
                            scan_rgb_dt_s=snapshot.scan_rgb_dt_s,
                            now_monotonic=plan_ready,
                        ),
                    }
                )
                self.get_logger().info(
                    "installed upstream NavDP trajectory: "
                    "points=%d mpc_horizon=%d laser_side=%s "
                    "clearance=%.3f->%.3f max_offset=%.3f"
                    % (
                        len(active_traj),
                        self.mpc.N,
                        laser_result.adjustment.side,
                        laser_result.adjustment.min_clearance_before_m,
                        laser_result.adjustment.min_clearance_after_m,
                        laser_result.adjustment.max_offset_m,
                    )
                )

            elapsed = time.monotonic() - cycle_start
            self.stop_event.wait(max(0.0, self.args.plan_period - elapsed))

    def _queue_visualization(self, snapshot: FrameSnapshot) -> None:
        request = VisualizationRequest(
            snapshot=snapshot, state=self.visualization_state
        )
        put_latest(
            self.visualization_queue,
            request,
        )

    def _visualization_loop(self) -> None:
        period = 1.0 / self.args.visualization_fps
        while not self.stop_event.is_set():
            try:
                request = self.visualization_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            cycle_start = time.monotonic()
            self._render_and_publish_visualization(request)
            elapsed = time.monotonic() - cycle_start
            self.stop_event.wait(max(0.0, period - elapsed))

    def _render_and_publish_visualization(
        self,
        request: VisualizationRequest,
    ) -> None:
        snapshot = request.snapshot
        state = request.state
        if state is None or snapshot.odom_xy_yaw is None:
            self._append_mpc_bev_video(snapshot)
            return
        try:
            rgb_bgr, depth_m, intrinsic = resize_rgbd_for_visualization(
                snapshot.rgb_bgr,
                snapshot.depth_m,
                snapshot.intrinsic,
                target_width=self.args.visualization_width,
            )
            goal_bgr = cv2.resize(
                self.goal_bgr,
                (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            rgb_and_goal = np.concatenate((rgb_bgr, goal_bgr), axis=1)
            visualization = self.visualizer.visualize_trajectory(
                rgb_and_goal,
                depth_m[:, :, None],
                intrinsic,
                state.trajectory,
                robot_pose=snapshot.odom_xy_yaw,
                all_trajectories_points=state.candidates,
                all_trajectories_values=state.values,
                show_depth=False,
                trajectory_prefix_points=state.trajectory_prefix,
            )
            status = "ARRIVED" if state.arrived else "NAVIGATING"
            cv2.rectangle(visualization, (0, 0), (520, 44), (0, 0, 0), -1)
            cv2.putText(
                visualization,
                f"{status}  critic max: {state.critic:.3f}",
                (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if not state.arrived else (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            message = self.bridge.cv2_to_imgmsg(visualization, encoding="bgr8")
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "odom"
            self.visualization_pub.publish(message)

            self.visualization_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.visualization_output.with_name(
                f"{self.visualization_output.stem}.tmp"
                f"{self.visualization_output.suffix or '.jpg'}"
            )
            if not cv2.imwrite(str(temporary), visualization):
                raise RuntimeError(f"cannot write visualization: {temporary}")
            temporary.replace(self.visualization_output)
            self._append_visualization_video(visualization)
        except Exception as error:
            self.get_logger().error(f"visualization failed: {error}")
        self._append_mpc_bev_video(snapshot)

    def _append_visualization_video(self, visualization: np.ndarray) -> None:
        if self.visualization_video_writer is None:
            self.visualization_video_output.parent.mkdir(parents=True, exist_ok=True)
            self.visualization_video_temporary.unlink(missing_ok=True)
            height, width = visualization.shape[:2]
            self.visualization_video_writer = cv2.VideoWriter(
                str(self.visualization_video_temporary),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.args.visualization_fps,
                (width, height),
            )
            if not self.visualization_video_writer.isOpened():
                self.visualization_video_writer.release()
                self.visualization_video_writer = None
                raise RuntimeError(
                    f"cannot open visualization video: {self.visualization_video_temporary}"
                )
            self.get_logger().info(
                f"visualization MP4 recording: {self.visualization_video_output}"
            )
        self.visualization_video_writer.write(visualization)

    def _render_mpc_bev(self, snapshot: FrameSnapshot) -> np.ndarray:
        frame_odom = (
            None
            if snapshot.odom_xy_yaw is None
            else snapshot.odom_xy_yaw.copy()
        )
        with self.data_lock:
            last_odom_time = self.last_odom_time
            odom_history = np.asarray(
                list(self.odom_history),
                dtype=np.float64,
            ).reshape(-1, 3)
            actual_velocity = (
                None
                if self.latest_odom_twist is None
                else self.latest_odom_twist.copy()
            )
            mpc_snapshot = self.latest_mpc_visualization
            latest_scan = self.latest_scan

        now = time.monotonic()
        try:
            laser_obstacle_xy, laser_status, laser_age_s = (
                laser_bev_obstacles(
                    latest_scan,
                    target_odom_xy_yaw=frame_odom,
                    now_monotonic=now,
                    timeout_s=self.args.scan_timeout,
                    config=self.laser_map_config,
                )
            )
        except Exception as error:
            laser_obstacle_xy = None
            laser_status = "LASER ERROR"
            laser_age_s = (
                None
                if latest_scan is None
                else now - latest_scan.received_at
            )
            if now - self.last_laser_map_error_log >= 2.0:
                self.get_logger().error(f"laser map rendering failed: {error}")
                self.last_laser_map_error_log = now
        mpc_updated_at = (
            None if mpc_snapshot is None else mpc_snapshot.updated_at
        )
        odom_status, mpc_fresh, mpc_status = bev_freshness(
            now=now,
            current_odom=frame_odom,
            last_odom_time=last_odom_time,
            mpc_updated_at=mpc_updated_at,
            odom_timeout=self.args.odom_timeout,
            mpc_timeout=self.args.mpc_bev_timeout,
        )
        predicted_states = None
        active_traj = None
        selected_diffusion = None
        command = np.zeros(2, dtype=np.float64)
        solve_ms = None
        if mpc_fresh:
            command = mpc_snapshot.command
            solve_ms = mpc_snapshot.solve_ms
        if mpc_fresh and odom_status == "ODOM OK":
            predicted_states = mpc_snapshot.predicted_states
            active_traj = mpc_snapshot.active_traj
            selected_diffusion = mpc_snapshot.selected_diffusion
        if odom_status != "ODOM OK":
            actual_velocity = None

        return render_mpc_rgb_bev(
            rgb_bgr=snapshot.rgb_bgr,
            depth_m=snapshot.depth_m,
            intrinsic=snapshot.intrinsic,
            base_from_camera=snapshot.base_from_camera,
            current_odom_xy_yaw=frame_odom,
            odom_history=odom_history,
            predicted_states=predicted_states,
            command=command,
            solve_ms=solve_ms,
            odom_status=odom_status,
            mpc_status=mpc_status,
            config=self.bev_config,
            actual_velocity=actual_velocity,
            active_traj=active_traj,
            selected_diffusion=selected_diffusion,
            laser_obstacle_xy=laser_obstacle_xy,
            laser_status=laser_status,
            laser_age_s=laser_age_s,
        )

    def _append_mpc_bev_video(self, snapshot: FrameSnapshot) -> None:
        if self.mpc_bev_video_failed:
            return
        try:
            frame = self._render_mpc_bev(snapshot)
            if self.mpc_bev_video_writer is None:
                self.mpc_bev_video_output.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                self.mpc_bev_video_temporary.unlink(missing_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(self.mpc_bev_video_temporary),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.args.visualization_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(
                        "cannot open MPC BEV video: "
                        f"{self.mpc_bev_video_temporary}"
                    )
                self.mpc_bev_video_writer = writer
                self.get_logger().info(
                    "MPC RGB BEV MP4 recording: "
                    f"{self.mpc_bev_video_output}"
                )
            self.mpc_bev_video_writer.write(frame)
        except Exception as error:
            self.mpc_bev_video_failed = True
            if self.mpc_bev_video_writer is not None:
                self.mpc_bev_video_writer.release()
                self.mpc_bev_video_writer = None
            self.mpc_bev_video_temporary.unlink(missing_ok=True)
            self.get_logger().error(f"MPC BEV recording disabled: {error}")

    def _control_loop(self) -> None:
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            with self.data_lock:
                latest_odom = (
                    None if self.latest_odom is None else self.latest_odom.copy()
                )
                latest_odom_twist = (
                    None
                    if self.latest_odom_twist is None
                    else self.latest_odom_twist.copy()
                )
                last_frame_time = (
                    None if self.latest_frame is None else self.latest_frame.received_at
                )
                frame_sequence = (
                    None if self.latest_frame is None else self.latest_frame.sequence
                )
                last_odom_time = self.last_odom_time
                last_plan_time = self.last_plan_time
                plan_id = self.latest_plan_id
                latest_scan = self.latest_scan
                reason = control_stop_reason(
                    now=cycle_start,
                    enable_control=self.args.enable_control,
                    arrival_blocked=self.arrival_blocked,
                    trajectory_ready=self.trajectory_ready,
                    last_frame_time=last_frame_time,
                    last_odom_time=last_odom_time,
                    last_plan_time=last_plan_time,
                    frame_timeout=self.args.frame_timeout,
                    odom_timeout=self.args.odom_timeout,
                    plan_timeout=self.args.plan_timeout,
                    last_scan_time=(
                        None
                        if latest_scan is None
                        else latest_scan.received_at
                    ),
                    scan_timeout=self.args.scan_timeout,
                )
                if (
                    reason is None
                    and self.local_nav_result_future is not None
                    and self.local_nav_result_future.done()
                ):
                    reason = "local_nav_finished"

            linear = 0.0
            angular = 0.0
            predicted_states = None
            reference_states = None
            solve_ms = None
            solve_started = None
            control_generation = None
            command_published = False
            if reason is None and latest_odom is not None:
                try:
                    with self.mpc_lock:
                        if self.mpc is None:
                            reason = "mpc_missing"
                        else:
                            with self.data_lock:
                                control_generation = self.trajectory_generation
                                plan_id = self.latest_plan_id
                                if self.arrival_blocked:
                                    reason = "arrival"
                                elif not self.trajectory_ready:
                                    reason = "tracking_invalidated"
                        if reason is None:
                            reference_states = self.mpc.find_reference_traj(
                                latest_odom,
                                self.mpc.ref_traj,
                            )
                            solve_started = time.perf_counter()
                            controls, predicted_states = self.mpc.solve(latest_odom)
                            solve_ms = (
                                time.perf_counter() - solve_started
                            ) * 1000.0
                            active_traj_snapshot = self.installed_active_traj.copy()
                            active_traj_snapshot.setflags(write=False)
                            selected_diffusion_snapshot = (
                                None
                                if self.selected_diffusion_state.installed is None
                                else self.selected_diffusion_state.installed.copy()
                            )
                            if selected_diffusion_snapshot is not None:
                                selected_diffusion_snapshot.setflags(write=False)
                            proposed_linear = float(
                                np.clip(
                                    controls[0, 0],
                                    0.0,
                                    self.control_max_v,
                                )
                            )
                            proposed_angular = float(
                                np.clip(
                                    controls[0, 1],
                                    -self.args.max_w,
                                    self.args.max_w,
                                )
                            )
                            predicted_snapshot = np.asarray(
                                predicted_states
                            ).copy()
                            predicted_snapshot.setflags(write=False)
                            command_snapshot = np.array(
                                [proposed_linear, proposed_angular],
                                dtype=np.float64,
                            )
                            command_snapshot.setflags(write=False)
                            with self.data_lock:
                                if self.arrival_blocked:
                                    reason = "arrival"
                                elif not tracking_generation_is_current(
                                    captured_generation=control_generation,
                                    current_generation=self.trajectory_generation,
                                    trajectory_ready=self.trajectory_ready,
                                ):
                                    reason = "tracking_invalidated"
                                if reason is None:
                                    linear = proposed_linear
                                    angular = proposed_angular
                                    self.latest_mpc_visualization = (
                                        MpcVisualizationSnapshot(
                                            predicted_states=predicted_snapshot,
                                            active_traj=active_traj_snapshot,
                                            selected_diffusion=selected_diffusion_snapshot,
                                            command=command_snapshot,
                                            solve_ms=float(solve_ms),
                                            updated_at=time.monotonic(),
                                        )
                                    )
                                self._publish_velocity(linear, angular)
                                command_published = True
                except Exception as error:
                    if solve_started is not None:
                        solve_ms = (
                            time.perf_counter() - solve_started
                        ) * 1000.0
                    reason = "mpc_error"
                    self.get_logger().error(f"MPC solve failed: {error}")

            if not command_published:
                with self.data_lock:
                    self._publish_velocity(linear, angular)
            self._write_diagnostic(
                {
                    "type": "control",
                    "wall_time": time.time(),
                    "monotonic_time": cycle_start,
                    "plan_id": plan_id,
                    "frame_sequence": frame_sequence,
                    "mode": "enabled" if self.args.enable_control else "dry-run",
                    "reason": "tracking" if reason is None else reason,
                    "odom": latest_odom,
                    "odom_twist": latest_odom_twist,
                    "command": [linear, angular],
                    "desired_velocity": [linear, angular],
                    "actual_velocity": latest_odom_twist,
                    "reference_states": reference_states,
                    "predicted_states": predicted_states,
                    "solve_ms": solve_ms,
                    "frame_age_s": None
                    if last_frame_time is None
                    else cycle_start - last_frame_time,
                    "odom_age_s": None
                    if last_odom_time is None
                    else cycle_start - last_odom_time,
                    "plan_age_s": None
                    if last_plan_time is None
                    else cycle_start - last_plan_time,
                    "scan_sequence": (
                        None if latest_scan is None else latest_scan.sequence
                    ),
                    "scan_age_s": (
                        None
                        if latest_scan is None
                        else cycle_start - latest_scan.received_at
                    ),
                }
            )
            if (
                reason != self.last_control_reason
                or cycle_start - self.last_control_log >= 1.0
            ):
                self.get_logger().info(
                    "control: mode=%s reason=%s v=%.3f w=%.3f"
                    % (
                        "enabled" if self.args.enable_control else "dry-run",
                        "tracking" if reason is None else reason,
                        linear,
                        angular,
                    )
                )
                self.last_control_reason = reason
                self.last_control_log = cycle_start
            self.stop_event.wait(max(0.0, 0.1 - (time.monotonic() - cycle_start)))

    def _publish_velocity(self, linear: float, angular: float) -> None:
        if not self.args.enable_control:
            return
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self.args.base_frame
        command.twist.linear.x = float(
            np.clip(linear, 0.0, self.control_max_v)
        )
        command.twist.angular.z = float(
            np.clip(angular, -self.args.max_w, self.args.max_w)
        )
        self.control_pub.publish(command)

    def stop(self) -> None:
        self.stop_event.set()
        if self.args.enable_control:
            for _ in range(3):
                self._publish_velocity(0.0, 0.0)
                time.sleep(0.05)
        self._cancel_posture()
        self._cancel_local_nav()
        if self.planning_thread.ident is not None:
            self.planning_thread.join(timeout=1.0)
        if self.control_thread.ident is not None:
            self.control_thread.join(timeout=1.0)
        if self.visualization_thread.ident is not None:
            self.visualization_thread.join()
        self.mpc_diagnostics.close()
        if self.mpc_bev_video_writer is not None:
            try:
                self.mpc_bev_video_writer.release()
                self.mpc_bev_video_writer = None
                finalize_mp4(
                    self.mpc_bev_video_temporary,
                    self.mpc_bev_video_output,
                )
                self.get_logger().info(
                    "MPC RGB BEV MP4 finalized (H.264): "
                    f"{self.mpc_bev_video_output}"
                )
            except Exception as error:
                self.mpc_bev_video_writer = None
                self.get_logger().error(
                    f"MPC BEV finalization failed: {error}"
                )
        if self.visualization_video_writer is not None:
            try:
                self.visualization_video_writer.release()
                self.visualization_video_writer = None
                finalize_mp4(
                    self.visualization_video_temporary,
                    self.visualization_video_output,
                )
                self.get_logger().info(
                    "visualization MP4 finalized (H.264): "
                    f"{self.visualization_video_output}"
                )
            except Exception as error:
                self.visualization_video_writer = None
                self.get_logger().error(
                    f"visualization finalization failed: {error}"
                )
        if self.server_initialized:
            try:
                result = navigator_close(port=self.args.server_port)
                self.server_initialized = False
                self.get_logger().info(
                    f"NavDP closed: status={result.get('status', 'unknown')}"
                )
            except Exception as error:
                self.get_logger().warning(f"NavDP close failed: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NavDP ImageGoal ROS2 wheeled client")
    parser.add_argument("--goal-image", required=True)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument("--server-port", type=int, default=8888)
    parser.add_argument("--rgb-topic", default="/cam_head/d435/color/image_raw")
    parser.add_argument(
        "--depth-topic",
        default="/cam_head/d435/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/cam_head/d435/color/camera_info",
    )
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--posture-timeout", type=float, default=10.0)
    parser.add_argument("--local-nav-timeout", type=float, default=5.0)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="d435_color_optical_frame")
    parser.add_argument("--laser-frame", default="laser_frame")
    parser.add_argument("--laser-x", type=float, default=0.042)
    parser.add_argument("--laser-y", type=float, default=0.0)
    parser.add_argument(
        "--laser-yaw",
        type=float,
        default=MIRA3_LASER_YAW_RAD,
    )
    parser.add_argument("--tf-timeout", type=float, default=0.2)
    parser.add_argument("--visualization-topic", default="/navdp/visualization")
    parser.add_argument(
        "--visualization-output",
        default="latest_visualization.jpg",
    )
    parser.add_argument(
        "--visualization-video-dir",
        default="~/NavDP-official-bebb436/navdp_visualizations",
    )
    parser.add_argument(
        "--mpc-log-dir",
        default="~/NavDP-official-bebb436/navdp_logs",
    )
    parser.add_argument("--visualization-width", type=int, default=320)
    parser.add_argument("--visualization-fps", type=float, default=15.0)
    parser.add_argument("--bev-sample-stride", type=int, default=4)
    parser.add_argument("--mpc-bev-timeout", type=float, default=0.5)
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--sync-slop", type=float, default=0.1)
    parser.add_argument(
        "--scan-sync-slop",
        type=float,
        default=DEFAULT_SCAN_SYNC_SLOP_S,
    )
    parser.add_argument("--scan-timeout", type=float, default=0.25)
    parser.add_argument("--laser-map-resolution", type=float, default=0.05)
    parser.add_argument("--plan-period", type=float, default=0.3)
    parser.add_argument("--max-v", type=float, default=0.1)
    parser.add_argument("--max-w", type=float, default=0.20)
    parser.add_argument("--critic-threshold", type=float, default=-3.0)
    parser.add_argument("--arrival-distance", type=float, default=0.5)
    parser.add_argument("--arrival-consecutive", type=int, default=3)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--frame-timeout", type=float, default=1.0)
    parser.add_argument("--odom-timeout", type=float, default=0.5)
    parser.add_argument("--plan-timeout", type=float, default=1.5)
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


GRACEFUL_SHUTDOWN_SIGNALS = (
    signal.SIGINT,
    signal.SIGTERM,
    signal.SIGHUP,
)


def _install_graceful_shutdown_handlers() -> None:
    for signum in GRACEFUL_SHUTDOWN_SIGNALS:
        signal.signal(signum, signal.default_int_handler)


def _stop_ignoring_shutdown_signals(node: NavdpImageGoalClient) -> None:
    previous_handlers = {
        signum: signal.signal(signum, signal.SIG_IGN)
        for signum in GRACEFUL_SHUTDOWN_SIGNALS
    }
    try:
        node.stop()
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(args.opencv_threads)
    _install_graceful_shutdown_handlers()
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = None
    try:
        node = NavdpImageGoalClient(args)
        node.start()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            if node is not None:
                executor.remove_node(node)
            executor.shutdown()
        if node is not None:
            _stop_ignoring_shutdown_signals(node)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
