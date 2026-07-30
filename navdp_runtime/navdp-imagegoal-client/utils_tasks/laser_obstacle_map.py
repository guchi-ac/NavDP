from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LaserMapConfig:
    forward_m: float = 6.0
    rear_m: float = 2.0
    lateral_m: float = 4.0
    resolution_m: float = 0.05
    laser_x_m: float = 0.042
    laser_y_m: float = 0.0
    laser_yaw_rad: float = 0.0
    self_half_length_m: float = 0.255
    self_half_width_m: float = 0.260

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.forward_m,
                self.rear_m,
                self.lateral_m,
                self.resolution_m,
                self.laser_x_m,
                self.laser_y_m,
                self.laser_yaw_rad,
                self.self_half_length_m,
                self.self_half_width_m,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("laser map configuration must be finite")
        if min(
            self.forward_m,
            self.rear_m,
            self.lateral_m,
            self.resolution_m,
            self.self_half_length_m,
            self.self_half_width_m,
        ) <= 0.0:
            raise ValueError("laser map dimensions must be positive")


@dataclass(frozen=True)
class LaserScanSnapshot:
    sequence: int
    stamp_ns: int
    received_at: float
    frame_id: str
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray
    odom_xy_yaw: Optional[np.ndarray]


@dataclass(frozen=True)
class LaserObstacleMap:
    grid: np.ndarray
    obstacle_xy: np.ndarray
    valid_count: int
    invalid_count: int
    self_filtered_count: int


def make_laser_scan_snapshot(
    *,
    sequence: int,
    stamp_ns: int,
    received_at: float,
    frame_id: str,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    ranges: np.ndarray,
    odom_xy_yaw: Optional[np.ndarray],
) -> LaserScanSnapshot:
    if not isinstance(sequence, (int, np.integer)) or sequence <= 0:
        raise ValueError("sequence must be a positive integer")
    if not isinstance(stamp_ns, (int, np.integer)) or stamp_ns < 0:
        raise ValueError("stamp_ns must be a non-negative integer")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id must be non-empty")

    metadata = np.asarray(
        [received_at, angle_min, angle_increment, range_min, range_max],
        dtype=np.float64,
    )
    if not np.isfinite(metadata).all():
        raise ValueError("scan metadata must be finite")
    if angle_increment == 0.0:
        raise ValueError("angle_increment must be nonzero")
    if range_min < 0.0 or range_min >= range_max:
        raise ValueError("range bounds must satisfy 0 <= min < max")

    owned_ranges = np.asarray(ranges, dtype=np.float32)
    if owned_ranges.ndim != 1 or owned_ranges.size == 0:
        raise ValueError("ranges must be a non-empty one-dimensional array")
    owned_ranges = owned_ranges.copy()
    owned_ranges.setflags(write=False)

    owned_odom = None
    if odom_xy_yaw is not None:
        owned_odom = np.asarray(odom_xy_yaw, dtype=np.float64)
        if owned_odom.shape != (3,) or not np.isfinite(owned_odom).all():
            raise ValueError("odom_xy_yaw must be a finite three-vector")
        owned_odom = owned_odom.copy()
        owned_odom.setflags(write=False)

    return LaserScanSnapshot(
        sequence=int(sequence),
        stamp_ns=int(stamp_ns),
        received_at=float(received_at),
        frame_id=frame_id,
        angle_min=float(angle_min),
        angle_increment=float(angle_increment),
        range_min=float(range_min),
        range_max=float(range_max),
        ranges=owned_ranges,
        odom_xy_yaw=owned_odom,
    )


def nearest_scan_snapshot(
    scans: Sequence[LaserScanSnapshot],
    target_stamp_ns: int,
    max_delta_s: float,
) -> Tuple[Optional[LaserScanSnapshot], Optional[float]]:
    if not np.isfinite(max_delta_s) or max_delta_s < 0.0:
        raise ValueError("max_delta_s must be finite and non-negative")
    if not scans:
        return None, None

    selected = min(scans, key=lambda scan: abs(scan.stamp_ns - target_stamp_ns))
    delta_s = (selected.stamp_ns - target_stamp_ns) / 1e9
    if abs(delta_s) > max_delta_s + np.finfo(np.float64).eps:
        return None, None
    return selected, delta_s


def laser_scan_record(
    snapshot: LaserScanSnapshot,
    *,
    wall_time: float,
) -> dict:
    return {
        "type": "scan",
        "wall_time": float(wall_time),
        "monotonic_time": snapshot.received_at,
        "scan_sequence": snapshot.sequence,
        "stamp_ns": snapshot.stamp_ns,
        "frame_id": snapshot.frame_id,
        "angle_min": snapshot.angle_min,
        "angle_increment": snapshot.angle_increment,
        "range_min": snapshot.range_min,
        "range_max": snapshot.range_max,
        "ranges": snapshot.ranges,
        "odom": snapshot.odom_xy_yaw,
    }


def laser_scan_association(
    snapshot: Optional[LaserScanSnapshot],
    *,
    scan_rgb_dt_s: Optional[float],
    now_monotonic: float,
) -> dict:
    if snapshot is None:
        return {
            "scan_sequence": None,
            "scan_stamp_ns": None,
            "scan_rgb_dt_s": None,
            "scan_age_s": None,
        }
    return {
        "scan_sequence": snapshot.sequence,
        "scan_stamp_ns": snapshot.stamp_ns,
        "scan_rgb_dt_s": scan_rgb_dt_s,
        "scan_age_s": float(now_monotonic) - snapshot.received_at,
    }


def build_current_obstacle_map(
    snapshot: LaserScanSnapshot,
    config: LaserMapConfig,
) -> LaserObstacleMap:
    ranges = snapshot.ranges.astype(np.float64, copy=False)
    angles = (
        snapshot.angle_min
        + np.arange(ranges.size, dtype=np.float64) * snapshot.angle_increment
    )
    range_valid = (
        np.isfinite(ranges)
        & (ranges != 0.0)
        & (ranges >= snapshot.range_min)
        & (ranges <= snapshot.range_max)
    )
    invalid_count = int(ranges.size - np.count_nonzero(range_valid))

    accepted_ranges = ranges[range_valid]
    accepted_angles = angles[range_valid]
    laser_x = accepted_ranges * np.cos(accepted_angles)
    laser_y = accepted_ranges * np.sin(accepted_angles)
    c = np.cos(config.laser_yaw_rad)
    s = np.sin(config.laser_yaw_rad)
    base_x = config.laser_x_m + c * laser_x - s * laser_y
    base_y = config.laser_y_m + s * laser_x + c * laser_y

    self_mask = (
        (np.abs(base_x) <= config.self_half_length_m)
        & (np.abs(base_y) <= config.self_half_width_m)
    )
    self_filtered_count = int(np.count_nonzero(self_mask))
    base_x = base_x[~self_mask]
    base_y = base_y[~self_mask]

    inside_bev = (
        (base_x >= -config.rear_m)
        & (base_x <= config.forward_m)
        & (base_y >= -config.lateral_m)
        & (base_y <= config.lateral_m)
    )
    base_x = base_x[inside_bev]
    base_y = base_y[inside_bev]

    height = round((config.forward_m + config.rear_m) / config.resolution_m)
    width = round((2.0 * config.lateral_m) / config.resolution_m)
    grid = np.zeros((height, width), dtype=np.uint8)

    rows = np.rint((config.forward_m - base_x) / config.resolution_m).astype(
        np.int64
    )
    cols = np.rint((config.lateral_m - base_y) / config.resolution_m).astype(
        np.int64
    )
    cells_inside = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )
    rows = rows[cells_inside]
    cols = cols[cells_inside]
    base_x = base_x[cells_inside]
    base_y = base_y[cells_inside]
    grid[rows, cols] = 100

    obstacle_xy = np.column_stack((base_x, base_y))
    grid.setflags(write=False)
    obstacle_xy.setflags(write=False)
    return LaserObstacleMap(
        grid=grid,
        obstacle_xy=obstacle_xy,
        valid_count=int(obstacle_xy.shape[0]),
        invalid_count=invalid_count,
        self_filtered_count=self_filtered_count,
    )


def laser_points_in_target_base(
    points_in_source_base: np.ndarray,
    source_odom_xy_yaw: Optional[np.ndarray],
    target_odom_xy_yaw: Optional[np.ndarray],
) -> np.ndarray:
    points = np.asarray(points_in_source_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_in_source_base must have shape (N, 2)")
    if source_odom_xy_yaw is None or target_odom_xy_yaw is None:
        return points.copy()

    source = np.asarray(source_odom_xy_yaw, dtype=np.float64)
    target = np.asarray(target_odom_xy_yaw, dtype=np.float64)
    if (
        source.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("odom poses must be finite three-vectors")

    source_c = np.cos(source[2])
    source_s = np.sin(source[2])
    world_x = source[0] + source_c * points[:, 0] - source_s * points[:, 1]
    world_y = source[1] + source_s * points[:, 0] + source_c * points[:, 1]

    delta_x = world_x - target[0]
    delta_y = world_y - target[1]
    target_c = np.cos(target[2])
    target_s = np.sin(target[2])
    target_x = target_c * delta_x + target_s * delta_y
    target_y = -target_s * delta_x + target_c * delta_y
    return np.column_stack((target_x, target_y))


def laser_bev_obstacles(
    snapshot: Optional[LaserScanSnapshot],
    *,
    target_odom_xy_yaw: Optional[np.ndarray],
    now_monotonic: float,
    timeout_s: float,
    config: LaserMapConfig,
) -> Tuple[Optional[np.ndarray], str, Optional[float]]:
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    if snapshot is None:
        return None, "LASER WAITING", None

    age_s = float(now_monotonic) - snapshot.received_at
    if age_s > timeout_s:
        return None, "LASER STALE", age_s

    obstacle_map = build_current_obstacle_map(snapshot, config)
    obstacle_xy = laser_points_in_target_base(
        obstacle_map.obstacle_xy,
        snapshot.odom_xy_yaw,
        target_odom_xy_yaw,
    )
    return (
        obstacle_xy,
        f"LASER OK points={obstacle_map.valid_count}",
        age_s,
    )
