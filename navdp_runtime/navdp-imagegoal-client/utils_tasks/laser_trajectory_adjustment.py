from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AdjustmentConfig:
    safety_radius_m: float = 0.35
    max_offset_m: float = 0.60
    offset_step_m: float = 0.02
    transition_m: float = 0.80

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.safety_radius_m,
                self.max_offset_m,
                self.offset_step_m,
                self.transition_m,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(
                "adjustment configuration must be finite and positive"
            )
        if self.offset_step_m > self.max_offset_m:
            raise ValueError("offset step must not exceed maximum offset")


@dataclass(frozen=True)
class TrajectoryAdjustment:
    original_xy: np.ndarray
    adjusted_xy: np.ndarray
    collision_mask: np.ndarray
    offsets_m: np.ndarray
    adjusted: bool
    safe: bool
    side: str
    min_clearance_before_m: float
    min_clearance_after_m: float
    max_offset_m: float


def _points_array(value, *, name: str, minimum_count: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < minimum_count
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            f"{name} must be finite with shape (N, 2), N >= {minimum_count}"
        )
    return points.copy()


def _trajectory_normals(points: np.ndarray) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > np.finfo(np.float64).eps
    if not np.any(valid):
        raise ValueError("dense trajectory must contain two distinct points")
    valid_indices = np.flatnonzero(valid)
    segment_tangents = deltas[valid] / lengths[valid, np.newaxis]
    point_indices = np.arange(len(points))
    nearest_segment = np.abs(
        point_indices[:, np.newaxis] - valid_indices[np.newaxis, :]
    ).argmin(axis=1)
    tangents = segment_tangents[nearest_segment]
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def _required_offsets(
    points: np.ndarray,
    normals: np.ndarray,
    collision_mask: np.ndarray,
    obstacle_tree: cKDTree,
    side_sign: float,
    config: AdjustmentConfig,
) -> np.ndarray | None:
    sample_offsets = np.arange(
        config.offset_step_m,
        config.max_offset_m + 0.5 * config.offset_step_m,
        config.offset_step_m,
        dtype=np.float64,
    )
    colliding_points = points[collision_mask]
    colliding_normals = normals[collision_mask]
    candidates = (
        colliding_points[:, np.newaxis, :]
        + side_sign
        * sample_offsets[np.newaxis, :, np.newaxis]
        * colliding_normals[:, np.newaxis, :]
    )
    distances = obstacle_tree.query(candidates.reshape(-1, 2))[0]
    safe_samples = distances.reshape(candidates.shape[:2]) >= (
        config.safety_radius_m - np.finfo(np.float64).eps
    )
    feasible = np.any(safe_samples, axis=1)
    if not np.all(feasible):
        return None

    required = np.zeros(len(points), dtype=np.float64)
    required[collision_mask] = sample_offsets[np.argmax(safe_samples, axis=1)]
    return required


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.asarray(value).copy()
    copied.setflags(write=False)
    return copied


def _smoothstep_offset_envelope(
    points: np.ndarray,
    required_offsets: np.ndarray,
    transition_m: float,
) -> np.ndarray | None:
    arc_m = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
        )
    )
    offsets = np.zeros(len(points), dtype=np.float64)
    for index in np.flatnonzero(required_offsets > 0.0):
        collision_arc_m = arc_m[index]
        if collision_arc_m <= np.finfo(np.float64).eps:
            return None
        ramp_start_m = max(0.0, collision_arc_m - transition_m)
        progress = np.clip(
            (arc_m - ramp_start_m) / (collision_arc_m - ramp_start_m),
            0.0,
            1.0,
        )
        ramp = required_offsets[index] * (
            progress * progress * (3.0 - 2.0 * progress)
        )
        ramp[arc_m >= collision_arc_m] = required_offsets[index]
        offsets = np.maximum(offsets, ramp)
    return offsets


def adjust_dense_trajectory(
    dense_xy,
    obstacle_xy,
    config: AdjustmentConfig = AdjustmentConfig(),
) -> TrajectoryAdjustment:
    points = _points_array(
        dense_xy,
        name="dense_xy",
        minimum_count=2,
    )
    obstacles = _points_array(
        obstacle_xy,
        name="obstacle_xy",
        minimum_count=1,
    )
    normals = _trajectory_normals(points)
    obstacle_tree = cKDTree(obstacles)
    clearance_before = obstacle_tree.query(points)[0]
    collision_mask = clearance_before < config.safety_radius_m

    if not np.any(collision_mask):
        zeros = np.zeros(len(points), dtype=np.float64)
        return TrajectoryAdjustment(
            original_xy=_readonly_copy(points),
            adjusted_xy=_readonly_copy(points),
            collision_mask=_readonly_copy(collision_mask),
            offsets_m=_readonly_copy(zeros),
            adjusted=False,
            safe=True,
            side="none",
            min_clearance_before_m=float(np.min(clearance_before)),
            min_clearance_after_m=float(np.min(clearance_before)),
            max_offset_m=0.0,
        )

    options = []
    for side, side_sign in (("left", 1.0), ("right", -1.0)):
        offsets = _required_offsets(
            points,
            normals,
            collision_mask,
            obstacle_tree,
            side_sign,
            config,
        )
        if offsets is not None:
            offsets = _smoothstep_offset_envelope(
                points,
                offsets,
                config.transition_m,
            )
        if offsets is not None:
            adjusted_xy = (
                points + side_sign * offsets[:, np.newaxis] * normals
            )
            clearance_after = obstacle_tree.query(adjusted_xy)[0]
            safe = bool(
                np.all(
                    clearance_after
                    >= config.safety_radius_m - np.finfo(np.float64).eps
                )
            )
            options.append(
                (
                    not safe,
                    float(np.max(offsets)),
                    side,
                    offsets,
                    adjusted_xy,
                    clearance_after,
                )
            )

    if not options:
        return TrajectoryAdjustment(
            original_xy=_readonly_copy(points),
            adjusted_xy=_readonly_copy(points),
            collision_mask=_readonly_copy(collision_mask),
            offsets_m=_readonly_copy(np.zeros(len(points), dtype=np.float64)),
            adjusted=False,
            safe=False,
            side="none",
            min_clearance_before_m=float(np.min(clearance_before)),
            min_clearance_after_m=float(np.min(clearance_before)),
            max_offset_m=0.0,
        )

    unsafe, _, side, offsets, adjusted_xy, clearance_after = min(
        options,
        key=lambda option: option[:2],
    )
    return TrajectoryAdjustment(
        original_xy=_readonly_copy(points),
        adjusted_xy=_readonly_copy(adjusted_xy),
        collision_mask=_readonly_copy(collision_mask),
        offsets_m=_readonly_copy(offsets),
        adjusted=True,
        safe=not unsafe,
        side=side,
        min_clearance_before_m=float(np.min(clearance_before)),
        min_clearance_after_m=float(np.min(clearance_after)),
        max_offset_m=float(np.max(offsets)),
    )
