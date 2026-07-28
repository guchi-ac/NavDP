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
class TrajectoryUpdate:
    active_traj: Optional[np.ndarray]
    candidate_accepted: bool
    reason: str
    join_distance_m: Optional[float]
    blind_length_m: float
    blind_point_count: int
    diffusion_length_m: float
    diffusion_point_count: int
    mpc_prediction_steps: int
    remaining_length_m: float
    manager_update_ms: float


class TrajectoryManager:
    """Maintain a stable odometry-frame guide path for MPC."""

    def __init__(
        self,
        *,
        point_spacing: float = 0.05,
        join_distance: float = 0.50,
        join_heading_degrees: float = 60.0,
        min_remaining: float = 0.20,
    ):
        values = {
            "point_spacing": point_spacing,
            "join_distance": join_distance,
            "join_heading_degrees": join_heading_degrees,
            "min_remaining": min_remaining,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        self.point_spacing = float(point_spacing)
        self.join_distance = float(join_distance)
        self.join_heading = math.radians(float(join_heading_degrees))
        self.min_remaining = float(min_remaining)
        self._active_traj = None
        self._blind_length_m = 0.0
        self._blind_point_count = 0
        self._diffusion_length_m = 0.0
        self._diffusion_point_count = 0

    def update(
        self,
        chassis_xy,
        candidate_world_xy=None,
        candidate_eligible: bool = False,
    ) -> TrajectoryUpdate:
        update_started = time.perf_counter()
        chassis = np.asarray(chassis_xy, dtype=np.float64)
        if chassis.shape != (2,) or not np.isfinite(chassis).all():
            raise ValueError("chassis_xy must be a finite shape-(2,) point")

        history, had_history = self._advance_history(chassis)
        active = history
        accepted = False
        join_distance_m = None
        reason = "history_retained" if history is not None else "no_history"

        candidate = None
        if candidate_world_xy is not None:
            if not candidate_eligible:
                reason = "candidate_low_critic"
            else:
                try:
                    candidate = self._normalize_polyline(candidate_world_xy)
                except ValueError:
                    reason = "candidate_invalid"

        if candidate is not None:
            if history is None:
                blind_path = self._build_blind_path(
                    np.vstack((chassis, candidate[0]))
                )
                blind_guides = blind_path[1:-1]
                initialized = self._assemble_active_trajectory(
                    chassis,
                    blind_guides,
                    candidate,
                )
                if self._polyline_length(initialized) >= self.min_remaining:
                    active = initialized
                    accepted = True
                    reason = "initialized"
                    self._set_candidate_metadata(
                        blind_path,
                        blind_guides,
                        candidate,
                    )
                else:
                    reason = "candidate_too_short"
            else:
                history_cumulative = self._cumulative_lengths(history)
                (
                    join_distance_m,
                    segment_index,
                    join_point,
                    _,
                ) = self._projection_with_arc(
                    history,
                    history_cumulative,
                    candidate[0],
                )
                history_heading = (
                    history[segment_index + 1] - history[segment_index]
                )
                candidate_heading = candidate[1] - candidate[0]
                if join_distance_m > self.join_distance:
                    reason = "join_distance"
                else:
                    if join_distance_m <= self.point_spacing:
                        heading_deltas = [
                            self._heading_delta(
                                history_heading,
                                candidate_heading,
                            )
                        ]
                    else:
                        connector_heading = candidate[0] - join_point
                        heading_deltas = [
                            self._heading_delta(
                                history_heading,
                                connector_heading,
                            ),
                            self._heading_delta(
                                connector_heading,
                                candidate_heading,
                            ),
                        ]
                    if max(heading_deltas) > self.join_heading:
                        reason = "join_heading"
                    else:
                        prefix_points = list(history[: segment_index + 1])
                        if not np.allclose(
                            prefix_points[-1],
                            join_point,
                            rtol=0.0,
                            atol=np.finfo(np.float64).eps,
                        ):
                            prefix_points.append(join_point)
                        if not np.allclose(
                            prefix_points[-1],
                            candidate[0],
                            rtol=0.0,
                            atol=np.finfo(np.float64).eps,
                        ):
                            prefix_points.append(candidate[0])
                        blind_path = self._build_blind_path(
                            np.asarray(prefix_points)
                        )
                        blind_guides = blind_path[1:-1]
                        proposed = self._assemble_active_trajectory(
                            chassis,
                            blind_guides,
                            candidate,
                        )
                        if (
                            self._polyline_length(proposed)
                            < self.min_remaining
                        ):
                            reason = "candidate_too_short"
                        else:
                            active = proposed
                            accepted = True
                            reason = "candidate_replaced"
                            self._set_candidate_metadata(
                                blind_path,
                                blind_guides,
                                candidate,
                            )

        if active is None and had_history and candidate_world_xy is None:
            reason = "history_exhausted"
        self._active_traj = None if active is None else active.copy()
        if self._active_traj is None:
            self._clear_candidate_metadata()
        remaining = (
            0.0
            if self._active_traj is None
            else self._polyline_length(self._active_traj)
        )
        result_traj = (
            None if self._active_traj is None else self._active_traj.copy()
        )
        if result_traj is not None:
            result_traj.setflags(write=False)
        return TrajectoryUpdate(
            active_traj=result_traj,
            candidate_accepted=accepted,
            reason=reason,
            join_distance_m=join_distance_m,
            blind_length_m=self._blind_length_m,
            blind_point_count=self._blind_point_count,
            diffusion_length_m=self._diffusion_length_m,
            diffusion_point_count=self._diffusion_point_count,
            mpc_prediction_steps=(
                self._blind_point_count + self._diffusion_point_count
            ),
            remaining_length_m=remaining,
            manager_update_ms=(time.perf_counter() - update_started) * 1000.0,
        )

    @staticmethod
    def _normalize_polyline(points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            raise ValueError("trajectory must be finite with shape (N, 2), N >= 2")
        keep = np.concatenate(
            (
                np.array([True]),
                np.linalg.norm(np.diff(points, axis=0), axis=1)
                > np.finfo(np.float64).eps,
            )
        )
        normalized = points[keep]
        if len(normalized) < 2:
            raise ValueError("trajectory must contain two distinct points")
        return normalized

    @staticmethod
    def _polyline_length(points: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    @staticmethod
    def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
        return np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )

    def _densify_preserving_vertices(self, points: np.ndarray) -> np.ndarray:
        points = self._normalize_polyline(points)
        dense = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            segment = end - start
            length = float(np.linalg.norm(segment))
            for distance in np.arange(
                self.point_spacing,
                length,
                self.point_spacing,
            ):
                dense.append(start + segment * (distance / length))
            dense.append(end)
        return self._normalize_polyline(np.asarray(dense))

    def _build_blind_path(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        keep = np.concatenate(
            (
                np.array([True]),
                np.linalg.norm(np.diff(points, axis=0), axis=1)
                > np.finfo(np.float64).eps,
            )
        )
        distinct = points[keep]
        if len(distinct) == 1:
            return distinct
        return self._densify_preserving_vertices(distinct)

    @staticmethod
    def _assemble_active_trajectory(
        chassis: np.ndarray,
        blind_guides: np.ndarray,
        candidate: np.ndarray,
    ) -> np.ndarray:
        if (
            len(blind_guides) == 0
            and np.array_equal(chassis, candidate[0])
        ):
            return candidate.copy()
        return np.vstack((chassis, blind_guides, candidate))

    def _set_candidate_metadata(
        self,
        blind_path: np.ndarray,
        blind_guides: np.ndarray,
        candidate: np.ndarray,
    ):
        self._blind_length_m = self._polyline_length(blind_path)
        self._blind_point_count = len(blind_guides)
        self._diffusion_length_m = self._polyline_length(candidate)
        self._diffusion_point_count = len(candidate)

    def _clear_candidate_metadata(self):
        self._blind_length_m = 0.0
        self._blind_point_count = 0
        self._diffusion_length_m = 0.0
        self._diffusion_point_count = 0

    @staticmethod
    def _projection_with_arc(
        polyline: np.ndarray,
        cumulative: np.ndarray,
        point: np.ndarray,
    ):
        best = (math.inf, 0, polyline[0], 0.0)
        for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
            segment = end - start
            segment_length = float(np.linalg.norm(segment))
            fraction = float(
                np.clip(
                    np.dot(point - start, segment) / np.dot(segment, segment),
                    0.0,
                    1.0,
                )
            )
            projection = start + fraction * segment
            distance = float(np.linalg.norm(point - projection))
            if distance < best[0]:
                projection_arc = float(
                    cumulative[index] + fraction * segment_length
                )
                best = (distance, index, projection, projection_arc)
        return best

    @classmethod
    def _closest_projection(cls, polyline: np.ndarray, point: np.ndarray):
        distance, index, projection, _ = cls._projection_with_arc(
            polyline,
            cls._cumulative_lengths(polyline),
            point,
        )
        return distance, index, projection

    def _advance_history(self, chassis: np.ndarray):
        if self._active_traj is None:
            return None, False
        _, segment_index, projection = self._closest_projection(
            self._active_traj,
            chassis,
        )
        forward_length = float(
            np.linalg.norm(
                self._active_traj[segment_index + 1] - projection
            )
        )
        if segment_index + 1 < len(self._active_traj) - 1:
            forward_length += self._polyline_length(
                self._active_traj[segment_index + 1 :]
            )
        if forward_length < self.min_remaining:
            return None, True
        try:
            remainder = self._normalize_polyline(
                np.vstack(
                    (
                        chassis,
                        projection,
                        self._active_traj[segment_index + 1 :],
                    )
                )
            )
        except ValueError:
            return None, True
        remainder[0] = chassis
        return remainder, True

    @staticmethod
    def _heading_delta(first: np.ndarray, second: np.ndarray) -> float:
        first_angle = math.atan2(first[1], first[0])
        second_angle = math.atan2(second[1], second[0])
        return abs(
            math.atan2(
                math.sin(second_angle - first_angle),
                math.cos(second_angle - first_angle),
            )
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
) -> Optional[str]:
    if not enable_control:
        return "control_disabled"
    if arrival_blocked:
        return "arrival"
    if not trajectory_ready:
        return "trajectory_missing"
    for name, timestamp, timeout in (
        ("frame", last_frame_time, frame_timeout),
        ("odom", last_odom_time, odom_timeout),
        ("plan", last_plan_time, plan_timeout),
    ):
        if timestamp is None:
            return f"{name}_missing"
        if now - timestamp > timeout:
            return f"{name}_stale"
    return None
