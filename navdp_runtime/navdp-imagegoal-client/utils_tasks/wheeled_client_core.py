import json
import math
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np



@dataclass(frozen=True)
class SelectedDiffusionInstallState:
    installed: Optional[np.ndarray] = None
    pending: Optional[np.ndarray] = None

    @staticmethod
    def _readonly_copy(points) -> np.ndarray:
        result = np.asarray(points, dtype=np.float64).copy()
        result.setflags(write=False)
        return result

    def stage(
        self,
        candidate_world_xy,
        candidate_accepted: bool,
    ):
        if not candidate_accepted:
            return self
        return SelectedDiffusionInstallState(
            installed=self.installed,
            pending=self._readonly_copy(candidate_world_xy),
        )

    def commit(self):
        if self.pending is None:
            return self
        return SelectedDiffusionInstallState(installed=self.pending)

    def clear(self):
        return SelectedDiffusionInstallState()


def normalize_tracking_trajectory(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "trajectory must be finite with shape (N, 2), N >= 2"
        )
    if not np.any(
        np.linalg.norm(points - points[0], axis=1)
        > np.finfo(np.float64).eps
    ):
        raise ValueError("trajectory must contain two distinct points")
    return points


def tracking_generation_is_current(
    *,
    captured_generation: int,
    current_generation: int,
    trajectory_ready: bool,
) -> bool:
    return (
        trajectory_ready
        and captured_generation == current_generation
    )




def configure_navdp_posture_goal(goal):
    """Configure the enabled-startup posture for a MIRA3 Torso goal."""
    goal.work_mode = 0
    goal.input_mode = 0
    goal.torso_roll = 0.0
    goal.torso_height = 0.0
    goal.torso_yaw = 0.0
    goal.head_pitch = -19.7795845
    goal.head_yaw = 0.0
    # MIRA3 groups are torso=[height, yaw], head=[head_yaw, head_pitch].
    goal.torso_mask = [False, True]
    goal.head_mask = [True, True]
    goal.max_velocity = 0.1
    return goal


class PostureActionRunner:
    """Run and, when needed, synchronously cancel the startup posture goal."""

    def __init__(
        self,
        *,
        node,
        action_client,
        timeout: float,
        spin_until_future_complete,
        goal_factory,
        succeeded_status: int,
    ):
        self.node = node
        self.action_client = action_client
        self.timeout = timeout
        self.spin_until_future_complete = spin_until_future_complete
        self.goal_factory = goal_factory
        self.succeeded_status = succeeded_status
        self.goal_handle = None

    def run(self):
        if not self.action_client.wait_for_server(timeout_sec=self.timeout):
            raise RuntimeError(
                "posture action server unavailable: /Torso/torso_action_service"
            )

        goal = configure_navdp_posture_goal(self.goal_factory())
        goal_future = self.action_client.send_goal_async(goal)
        self.spin_until_future_complete(
            self.node,
            goal_future,
            timeout_sec=self.timeout,
        )
        if not goal_future.done():
            raise RuntimeError("timed out sending posture action goal")
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("posture action goal was rejected")

        self.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        self.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=self.timeout,
        )
        if not result_future.done():
            try:
                self.cancel()
            except RuntimeError as error:
                raise RuntimeError(
                    "timed out waiting for posture action result; "
                    f"cancellation failed: {error}"
                ) from error
            raise RuntimeError("timed out waiting for posture action result")

        result_response = result_future.result()
        self.goal_handle = None
        if result_response is None:
            raise RuntimeError("posture action returned no result")
        result = result_response.result
        result_message = getattr(result, "msg", "")
        if result_response.status != self.succeeded_status:
            detail = result_message or f"status={result_response.status}"
            raise RuntimeError(f"posture action did not succeed: {detail}")
        if not result.success:
            raise RuntimeError(
                "posture action reported failure: "
                f"{result_message or 'no result message'}"
            )
        return result

    def cancel(self) -> None:
        if self.goal_handle is None:
            return
        cancel_future = self.goal_handle.cancel_goal_async()
        self.spin_until_future_complete(
            self.node,
            cancel_future,
            timeout_sec=self.timeout,
        )
        if not cancel_future.done():
            raise RuntimeError("timed out canceling posture action goal")
        cancel_response = cancel_future.result()
        if cancel_response is None or not cancel_response.goals_canceling:
            raise RuntimeError("posture action goal cancellation was rejected")
        self.goal_handle = None


def run_navdp_startup(
    *,
    enable_control: bool,
    align_posture,
    publish_camera_tf,
    start_local_nav,
    start_workers,
) -> None:
    if enable_control:
        align_posture()
    publish_camera_tf()
    start_local_nav()
    start_workers()


def finalize_mp4(temporary: Path, output: Path) -> bool:
    temporary = Path(temporary)
    output = Path(output)
    transcoded = output.with_name(f"{output.stem}.h264.partial.mp4")
    last_error = None
    for attempt in range(3):
        transcoded.unlink(missing_ok=True)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(temporary),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(transcoded),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, subprocess.CalledProcessError) as caught:
            last_error = caught
            if attempt < 2:
                time.sleep(0.5)
            continue
        break
    else:
        transcoded.unlink(missing_ok=True)
        if isinstance(last_error, subprocess.CalledProcessError):
            stderr = (last_error.stderr or b"").decode(
                "utf-8",
                errors="replace",
            ).strip()
            detail = f"returncode={last_error.returncode}"
            if last_error.returncode < 0:
                detail += f" signal={-last_error.returncode}"
            if stderr:
                detail += f" stderr={stderr}"
        else:
            detail = str(last_error)
        raise RuntimeError(
            f"H.264 transcode failed after 3 attempts: {detail}"
        )

    transcoded.replace(output)
    temporary.unlink(missing_ok=True)
    return True


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, record: dict) -> None:
        line = json.dumps(
            _json_value(record),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        with self._lock:
            if self._file is None:
                raise RuntimeError("JSONL writer is closed")
            self._file.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def put_latest(work_queue: queue.Queue, value) -> None:
    while True:
        try:
            work_queue.put_nowait(value)
            return
        except queue.Full:
            try:
                work_queue.get_nowait()
            except queue.Empty:
                pass


def resize_rgbd_for_visualization(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    target_width: int,
):
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    height, width = rgb_bgr.shape[:2]
    if depth_m.shape[:2] != (height, width):
        raise ValueError("RGB and depth dimensions must match")

    target_height = max(1, round(height * target_width / width))
    resized_rgb = cv2.resize(
        rgb_bgr,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    resized_depth = cv2.resize(
        depth_m,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
    resized_intrinsic[0, :] *= target_width / width
    resized_intrinsic[1, :] *= target_height / height
    resized_intrinsic[2, :] = intrinsic[2, :]
    return resized_rgb, resized_depth, resized_intrinsic


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def camera_pose_from_transform(
    translation_xyz: np.ndarray,
    rotation_xyzw: np.ndarray,
) -> np.ndarray:
    translation_xyz = np.asarray(translation_xyz, dtype=np.float64)
    rotation_xyzw = np.asarray(rotation_xyzw, dtype=np.float64)
    if translation_xyz.shape != (3,):
        raise ValueError("translation_xyz must have shape (3,)")
    if rotation_xyzw.shape != (4,):
        raise ValueError("rotation_xyzw must have shape (4,)")
    transform = transform_matrix_from_translation_quaternion(
        translation_xyz,
        rotation_xyzw,
    )
    optical_forward_xy = transform[:2, 2]
    if np.linalg.norm(optical_forward_xy) <= np.finfo(np.float64).eps:
        raise ValueError("camera optical forward axis has no planar projection")
    return np.array(
        [
            translation_xyz[0],
            translation_xyz[1],
            math.atan2(optical_forward_xy[1], optical_forward_xy[0]),
        ],
        dtype=np.float64,
    )


def transform_matrix_from_translation_quaternion(
    translation_xyz: np.ndarray,
    rotation_xyzw: np.ndarray,
) -> np.ndarray:
    translation_xyz = np.asarray(translation_xyz, dtype=np.float64)
    rotation_xyzw = np.asarray(rotation_xyzw, dtype=np.float64)
    if translation_xyz.shape != (3,):
        raise ValueError("translation_xyz must have shape (3,)")
    if rotation_xyzw.shape != (4,):
        raise ValueError("rotation_xyzw must have shape (4,)")
    norm = np.linalg.norm(rotation_xyzw)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("rotation quaternion must be nonzero")
    x, y, z, w = rotation_xyzw / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_xyz
    return transform


NAVDP_OFFICIAL_CAMERA_HEIGHT_M = 0.2


def navdp_official_base_from_camera() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = np.array(
        [0.0, 0.0, NAVDP_OFFICIAL_CAMERA_HEIGHT_M],
        dtype=np.float64,
    )
    return transform


def navdp_virtual_pixels(
    local_xy: np.ndarray,
    intrinsic: np.ndarray,
    image_height: int,
    virtual_camera_height: float = 0.2,
) -> np.ndarray:
    local_xy = np.asarray(local_xy, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if local_xy.ndim != 2 or local_xy.shape[1] != 2 or len(local_xy) < 2:
        raise ValueError(
            "virtual_reprojection: local_xy must have shape (N, 2), N >= 2"
        )
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(
            "virtual_reprojection: intrinsic must be finite shape (3, 3)"
        )
    if (
        isinstance(image_height, (bool, np.bool_))
        or not isinstance(image_height, (int, np.integer))
        or image_height <= 0
    ):
        raise ValueError("virtual_reprojection: image_height must be positive")
    if not np.isfinite(virtual_camera_height) or virtual_camera_height <= 0.0:
        raise ValueError(
            "virtual_reprojection: virtual camera height must be positive"
        )
    if not np.isfinite(local_xy).all():
        raise ValueError("virtual_reprojection: local_xy must be finite")
    forward = local_xy[:, 0]
    if np.any(forward <= 0.0):
        raise ValueError(
            "virtual_reprojection: forward distance must be positive"
        )
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            "virtual_reprojection: focal lengths must be positive"
        )
    u = fx * (-local_xy[:, 1] / forward) + cx
    v = (
        float(image_height - 1)
        + fy * (virtual_camera_height / forward)
        - cy
    )
    return np.column_stack((u, v))


def reproject_navdp_to_ground_base(
    local_xy: np.ndarray,
    intrinsic: np.ndarray,
    image_height: int,
    base_from_camera: np.ndarray,
    virtual_camera_height: float = 0.2,
) -> np.ndarray:
    pixels = navdp_virtual_pixels(
        local_xy,
        intrinsic,
        image_height,
        virtual_camera_height,
    )
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    base_from_camera = np.asarray(base_from_camera, dtype=np.float64)
    if (
        base_from_camera.shape != (4, 4)
        or not np.isfinite(base_from_camera).all()
    ):
        raise ValueError(
            "virtual_reprojection: base_from_camera must be finite shape (4, 4)"
        )
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera_rays = np.column_stack(
        (
            (pixels[:, 0] - cx) / fx,
            (pixels[:, 1] - cy) / fy,
            np.ones(len(pixels)),
        )
    )
    base_rays = camera_rays @ base_from_camera[:3, :3].T
    camera_origin = base_from_camera[:3, 3]
    vertical = base_rays[:, 2]
    if np.any(np.abs(vertical) <= np.finfo(np.float64).eps):
        raise ValueError(
            "virtual_reprojection: ray is parallel to ground"
        )
    scales = -camera_origin[2] / vertical
    if np.any(scales <= 0.0):
        raise ValueError(
            "virtual_reprojection: ground intersection is behind camera"
        )
    base_points = camera_origin + scales[:, None] * base_rays
    base_xy = base_points[:, :2]
    if not np.isfinite(base_xy).all():
        raise ValueError(
            "virtual_reprojection: ground intersection must be finite"
        )
    if np.any(base_xy[:, 0] <= 0.0):
        raise ValueError(
            "virtual_reprojection: ground intersection must be forward"
        )
    return base_xy


def trajectory_to_world(
    local_xy: np.ndarray,
    odom_xy_yaw: np.ndarray,
    camera_x: float = 0.0,
    camera_y: float = 0.0,
    camera_yaw: float = 0.0,
) -> np.ndarray:
    local_xy = np.asarray(local_xy, dtype=np.float64)
    odom_xy_yaw = np.asarray(odom_xy_yaw, dtype=np.float64)
    if local_xy.ndim != 2 or local_xy.shape[1] != 2:
        raise ValueError("local_xy must have shape (N, 2)")
    if odom_xy_yaw.shape != (3,):
        raise ValueError("odom_xy_yaw must have shape (3,)")

    base_yaw = odom_xy_yaw[2]
    base_rotation = np.array(
        [
            [math.cos(base_yaw), -math.sin(base_yaw)],
            [math.sin(base_yaw), math.cos(base_yaw)],
        ]
    )
    trajectory_yaw = base_yaw + camera_yaw
    trajectory_rotation = np.array(
        [
            [math.cos(trajectory_yaw), -math.sin(trajectory_yaw)],
            [math.sin(trajectory_yaw), math.cos(trajectory_yaw)],
        ]
    )
    camera_world = odom_xy_yaw[:2] + base_rotation @ np.array([camera_x, camera_y])
    return camera_world + local_xy @ trajectory_rotation.T


def control_stop_reason(
    *,
    now: float,
    enable_control: bool,
    arrival_blocked: bool,
    trajectory_ready: bool,
    last_frame_time: Optional[float],
    last_odom_time: Optional[float],
    last_plan_time: Optional[float],
    frame_timeout: float,
    odom_timeout: float,
    plan_timeout: float,
    last_scan_time: Optional[float] = None,
    scan_timeout: Optional[float] = None,
) -> Optional[str]:
    if not enable_control:
        return "control_disabled"
    if arrival_blocked:
        return "arrival"
    if not trajectory_ready:
        return "trajectory_missing"
    timestamp_checks = [
        ("frame", last_frame_time, frame_timeout),
        ("odom", last_odom_time, odom_timeout),
        ("plan", last_plan_time, plan_timeout),
    ]
    if scan_timeout is not None:
        timestamp_checks.append(("scan", last_scan_time, scan_timeout))
    for name, timestamp, timeout in timestamp_checks:
        if timestamp is None:
            return f"{name}_missing"
        if now - timestamp > timeout:
            return f"{name}_stale"
    return None
