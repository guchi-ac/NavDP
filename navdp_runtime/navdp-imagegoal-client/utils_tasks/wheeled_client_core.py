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
    diffusion_spacing_m: float
    projection_advance_m: float
    blind_max_turn_degrees: float
    mpc_prediction_steps: int
    remaining_length_m: float
    manager_update_ms: float


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
        self._history_centerline = None
        self._diffusion_tail = None
        self._diffusion_start_arc = 0.0
        self._diffusion_spacing = self.point_spacing
        self._last_chassis = None
        self._active_traj = None
        self._blind_length_m = 0.0
        self._blind_point_count = 0
        self._diffusion_length_m = 0.0
        self._diffusion_point_count = 0
        self._blind_max_turn_degrees = 0.0

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

        had_history = self._history_centerline is not None
        history, projection_advance_m = self._advance_centerline(chassis)
        active = self._build_active_trajectory(chassis)
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
                spacing = self._candidate_spacing(
                    candidate,
                    self.point_spacing,
                )
                connector_length = float(
                    np.linalg.norm(candidate[0] - chassis)
                )
                proposed_centerline = self._normalize_polyline(
                    np.vstack((chassis, candidate))
                )
                proposed_active, blind_guides = self._active_from_state(
                    chassis,
                    proposed_centerline,
                    candidate,
                    connector_length,
                    spacing,
                )
                if not self._first_distinct_reference_is_forward(
                    chassis,
                    proposed_centerline,
                    proposed_active,
                ):
                    reason = "candidate_nonforward"
                elif (
                    self._polyline_length(proposed_active)
                    >= self.min_remaining
                ):
                    self._install_candidate_state(
                        proposed_centerline,
                        candidate,
                        connector_length,
                        spacing,
                    )
                    active = proposed_active
                    accepted = True
                    reason = "initialized"
                    self._set_candidate_metadata(
                        chassis,
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
                    join_arc,
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
                        history_prefix = self._prefix_through_arc(
                            history,
                            join_arc,
                        )
                        proposed_centerline = self._normalize_polyline(
                            np.vstack((history_prefix, candidate))
                        )
                        proposed_start_arc = (
                            self._polyline_length(history_prefix)
                            + float(
                                np.linalg.norm(
                                    candidate[0] - history_prefix[-1]
                                )
                            )
                        )
                        proposed_spacing = self._candidate_spacing(
                            candidate,
                            self.point_spacing,
                        )
                        proposed, blind_guides = self._active_from_state(
                            chassis,
                            proposed_centerline,
                            candidate,
                            proposed_start_arc,
                            proposed_spacing,
                        )
                        if not self._first_distinct_reference_is_forward(
                            chassis,
                            proposed_centerline,
                            proposed,
                        ):
                            reason = "candidate_nonforward"
                        elif (
                            self._polyline_length(proposed)
                            < self.min_remaining
                        ):
                            reason = "candidate_too_short"
                        else:
                            active = proposed
                            accepted = True
                            reason = "candidate_replaced"
                            self._install_candidate_state(
                                proposed_centerline,
                                candidate,
                                proposed_start_arc,
                                proposed_spacing,
                            )
                            self._set_candidate_metadata(
                                chassis,
                                blind_guides,
                                candidate,
                            )

        if active is None and had_history and candidate_world_xy is None:
            reason = "history_exhausted"
        self._active_traj = None if active is None else active.copy()
        if self._active_traj is None:
            self._clear_persistent_state()
        elif self._history_centerline is not None:
            self._last_chassis = chassis.copy()
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
            diffusion_spacing_m=(
                0.0
                if self._active_traj is None
                else self._diffusion_spacing
            ),
            projection_advance_m=projection_advance_m,
            blind_max_turn_degrees=self._blind_max_turn_degrees,
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

    @staticmethod
    def _candidate_spacing(candidate: np.ndarray, minimum: float) -> float:
        lengths = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
        return max(float(np.median(lengths)), minimum)

    @classmethod
    def _point_at_arc(cls, points: np.ndarray, target_arc: float):
        cumulative = cls._cumulative_lengths(points)
        target = float(np.clip(target_arc, 0.0, cumulative[-1]))
        index = min(
            int(np.searchsorted(cumulative, target, side="right") - 1),
            len(points) - 2,
        )
        segment_length = cumulative[index + 1] - cumulative[index]
        fraction = (target - cumulative[index]) / segment_length
        return index, points[index] + fraction * (
            points[index + 1] - points[index]
        )

    @classmethod
    def _trim_from_arc(cls, points: np.ndarray, start_arc: float) -> np.ndarray:
        index, boundary = cls._point_at_arc(points, start_arc)
        remainder = np.vstack((boundary, points[index + 1 :]))
        return cls._normalize_polyline(remainder)

    @classmethod
    def _prefix_through_arc(
        cls,
        points: np.ndarray,
        end_arc: float,
    ) -> np.ndarray:
        if end_arc <= np.finfo(np.float64).eps:
            return points[:1].copy()
        index, boundary = cls._point_at_arc(points, end_arc)
        prefix = np.vstack((points[: index + 1], boundary))
        keep = np.concatenate(
            (
                np.array([True]),
                np.linalg.norm(np.diff(prefix, axis=0), axis=1)
                > np.finfo(np.float64).eps,
            )
        )
        return prefix[keep]

    @classmethod
    def _sample_at_arcs(
        cls,
        points: np.ndarray,
        arcs: np.ndarray,
    ) -> np.ndarray:
        if len(arcs) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(
            [
                cls._point_at_arc(points, float(target_arc))[1]
                for target_arc in arcs
            ],
            dtype=np.float64,
        )

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

    def _active_from_state(
        self,
        chassis: np.ndarray,
        centerline: np.ndarray,
        candidate: np.ndarray,
        diffusion_start_arc: float,
        diffusion_spacing: float,
    ):
        blind_arcs = np.arange(
            diffusion_spacing,
            diffusion_start_arc,
            diffusion_spacing,
        )
        blind_guides = self._sample_at_arcs(centerline, blind_arcs)
        tangent = centerline[1] - centerline[0]
        blind_guides = np.asarray(
            [
                point
                for point in blind_guides
                if np.dot(point - chassis, tangent) > 0.0
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        active = self._assemble_active_trajectory(
            chassis,
            blind_guides,
            candidate,
        )
        return active, blind_guides

    @staticmethod
    def _first_distinct_reference_is_forward(
        chassis: np.ndarray,
        centerline: np.ndarray,
        active: np.ndarray,
    ) -> bool:
        tangent = centerline[1] - centerline[0]
        for point in active[1:]:
            if (
                np.linalg.norm(point - chassis)
                > np.finfo(np.float64).eps
            ):
                return bool(np.dot(point - chassis, tangent) > 0.0)
        return False

    def _build_active_trajectory(self, chassis: np.ndarray):
        if (
            self._history_centerline is None
            or self._diffusion_tail is None
        ):
            return None
        active, blind_guides = self._active_from_state(
            chassis,
            self._history_centerline,
            self._diffusion_tail,
            self._diffusion_start_arc,
            self._diffusion_spacing,
        )
        if not self._first_distinct_reference_is_forward(
            chassis,
            self._history_centerline,
            active,
        ):
            return None
        self._set_candidate_metadata(
            chassis,
            blind_guides,
            self._diffusion_tail,
        )
        return active

    def _install_candidate_state(
        self,
        centerline: np.ndarray,
        candidate: np.ndarray,
        diffusion_start_arc: float,
        diffusion_spacing: float,
    ):
        self._history_centerline = centerline.copy()
        self._diffusion_tail = candidate.copy()
        self._diffusion_start_arc = float(diffusion_start_arc)
        self._diffusion_spacing = float(diffusion_spacing)

    def _set_candidate_metadata(
        self,
        chassis: np.ndarray,
        blind_guides: np.ndarray,
        candidate: np.ndarray,
    ):
        blind_path = np.vstack((chassis, blind_guides, candidate[0]))
        self._blind_length_m = self._polyline_length(blind_path)
        self._blind_point_count = len(blind_guides)
        self._diffusion_length_m = self._polyline_length(candidate)
        self._diffusion_point_count = len(candidate)
        self._blind_max_turn_degrees = self._maximum_turn_degrees(
            blind_path
        )

    def _clear_persistent_state(self):
        self._history_centerline = None
        self._diffusion_tail = None
        self._diffusion_start_arc = 0.0
        self._diffusion_spacing = self.point_spacing
        self._last_chassis = None
        self._blind_length_m = 0.0
        self._blind_point_count = 0
        self._diffusion_length_m = 0.0
        self._diffusion_point_count = 0
        self._blind_max_turn_degrees = 0.0

    @classmethod
    def _maximum_turn_degrees(cls, points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        deltas = np.diff(points, axis=0)
        deltas = deltas[
            np.linalg.norm(deltas, axis=1)
            > np.finfo(np.float64).eps
        ]
        if len(deltas) < 2:
            return 0.0
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        turns = np.abs(
            np.arctan2(
                np.sin(np.diff(headings)),
                np.cos(np.diff(headings)),
            )
        )
        return math.degrees(float(np.max(turns)))

    @staticmethod
    def _projection_with_arc(
        polyline: np.ndarray,
        cumulative: np.ndarray,
        point: np.ndarray,
        maximum_arc: Optional[float] = None,
    ):
        best = (
            float(np.linalg.norm(point - polyline[0])),
            0,
            polyline[0],
            0.0,
        )
        for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
            segment = end - start
            segment_length = float(np.linalg.norm(segment))
            available_length = segment_length
            if maximum_arc is not None:
                available_length = min(
                    segment_length,
                    float(maximum_arc - cumulative[index]),
                )
                if available_length <= np.finfo(np.float64).eps:
                    break
                end = start + segment * (
                    available_length / segment_length
                )
                segment = end - start
            fraction = float(
                np.clip(
                    np.dot(point - start, segment) / np.dot(segment, segment),
                    0.0,
                    1.0,
                )
            )
            projection = start + fraction * segment
            distance = float(np.linalg.norm(point - projection))
            if distance < best[0] - 1e-9:
                projection_arc = float(
                    cumulative[index] + fraction * available_length
                )
                best = (distance, index, projection, projection_arc)
            if (
                maximum_arc is not None
                and cumulative[index] + available_length
                >= maximum_arc - 1e-9
            ):
                break
        return best

    def _advance_diffusion_suffix(self, projection_arc: float) -> bool:
        if self._diffusion_tail is None:
            return False
        if projection_arc <= self._diffusion_start_arc + 1e-9:
            self._diffusion_start_arc = max(
                0.0,
                self._diffusion_start_arc - projection_arc,
            )
            return True

        passed_in_diffusion = projection_arc - self._diffusion_start_arc
        candidate_cumulative = self._cumulative_lengths(
            self._diffusion_tail
        )
        keep_index = int(
            np.searchsorted(
                candidate_cumulative,
                passed_in_diffusion,
                side="right",
            )
        )
        if keep_index >= len(self._diffusion_tail):
            return False
        next_point_arc = float(candidate_cumulative[keep_index])
        self._diffusion_tail = self._diffusion_tail[keep_index:].copy()
        self._diffusion_start_arc = (
            next_point_arc - passed_in_diffusion
        )
        return True

    def _advance_centerline(self, chassis: np.ndarray):
        if self._history_centerline is None:
            return None, 0.0
        displacement = (
            0.0
            if self._last_chassis is None
            else float(np.linalg.norm(chassis - self._last_chassis))
        )
        maximum_arc = min(
            self._polyline_length(self._history_centerline),
            displacement + self._diffusion_spacing,
        )
        _, _, _, projection_arc = self._projection_with_arc(
            self._history_centerline,
            self._cumulative_lengths(self._history_centerline),
            chassis,
            maximum_arc=maximum_arc,
        )
        forward_length = (
            self._polyline_length(self._history_centerline)
            - projection_arc
        )
        if forward_length < self.min_remaining:
            self._clear_persistent_state()
            return None, projection_arc
        trimmed_centerline = self._trim_from_arc(
            self._history_centerline,
            projection_arc,
        )
        if not self._advance_diffusion_suffix(projection_arc):
            self._clear_persistent_state()
            return None, projection_arc
        self._history_centerline = trimmed_centerline
        return self._history_centerline, projection_arc

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
    if np.any(np.diff(base_xy[:, 0]) < -1e-6):
        raise ValueError(
            "virtual_reprojection: trajectory reverses forward progress"
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
