from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils_tasks.laser_obstacle_map import (
    LaserMapConfig,
    LaserScanSnapshot,
    build_current_obstacle_map,
    laser_points_in_target_base,
)
from utils_tasks.laser_trajectory_adjustment import (
    AdjustmentConfig,
    TrajectoryAdjustment,
    adjust_dense_trajectory,
)

LIVE_TEST_MAX_V_MPS = 0.1


@dataclass(frozen=True)
class LiveLaserTrajectoryResult:
    safe: bool
    reason: str
    trajectory_local_xy: Optional[np.ndarray]
    obstacle_local_xy: Optional[np.ndarray]
    adjustment: Optional[TrajectoryAdjustment]
    scan_sequence: Optional[int]
    scan_age_s: Optional[float]


def live_test_max_v(requested_max_v: float) -> float:
    requested_max_v = float(requested_max_v)
    if not np.isfinite(requested_max_v) or requested_max_v <= 0.0:
        raise ValueError("requested_max_v must be finite and positive")
    return min(requested_max_v, LIVE_TEST_MAX_V_MPS)


def _result(
    *,
    safe: bool,
    reason: str,
    trajectory_local_xy=None,
    obstacle_local_xy=None,
    adjustment=None,
    scan_sequence=None,
    scan_age_s=None,
) -> LiveLaserTrajectoryResult:
    return LiveLaserTrajectoryResult(
        safe=safe,
        reason=reason,
        trajectory_local_xy=trajectory_local_xy,
        obstacle_local_xy=obstacle_local_xy,
        adjustment=adjustment,
        scan_sequence=scan_sequence,
        scan_age_s=scan_age_s,
    )


def prepare_live_laser_trajectory(
    *,
    dense_xy,
    scan: Optional[LaserScanSnapshot],
    plan_odom_xy_yaw,
    now_monotonic: float,
    timeout_s: float,
    laser_config: LaserMapConfig = LaserMapConfig(),
    adjustment_config: AdjustmentConfig = AdjustmentConfig(),
) -> LiveLaserTrajectoryResult:
    if not np.isfinite(now_monotonic):
        raise ValueError("now_monotonic must be finite")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    plan_odom = np.asarray(plan_odom_xy_yaw, dtype=np.float64)
    if plan_odom.shape != (3,) or not np.isfinite(plan_odom).all():
        raise ValueError("plan_odom_xy_yaw must be a finite three-vector")

    if scan is None:
        return _result(safe=False, reason="scan_missing")

    scan_age_s = float(now_monotonic) - scan.received_at
    if scan_age_s > timeout_s:
        return _result(
            safe=False,
            reason="scan_stale",
            scan_sequence=scan.sequence,
            scan_age_s=scan_age_s,
        )
    if scan.odom_xy_yaw is None:
        return _result(
            safe=False,
            reason="scan_odom_missing",
            scan_sequence=scan.sequence,
            scan_age_s=scan_age_s,
        )

    obstacle_map = build_current_obstacle_map(scan, laser_config)
    if not len(obstacle_map.obstacle_xy):
        return _result(
            safe=False,
            reason="scan_empty",
            obstacle_local_xy=obstacle_map.obstacle_xy,
            scan_sequence=scan.sequence,
            scan_age_s=scan_age_s,
        )
    obstacle_local_xy = laser_points_in_target_base(
        obstacle_map.obstacle_xy,
        scan.odom_xy_yaw,
        plan_odom,
    )
    adjustment = adjust_dense_trajectory(
        dense_xy,
        obstacle_local_xy,
        adjustment_config,
    )
    if not adjustment.safe:
        return _result(
            safe=False,
            reason="adjustment_unsafe",
            obstacle_local_xy=obstacle_local_xy,
            adjustment=adjustment,
            scan_sequence=scan.sequence,
            scan_age_s=scan_age_s,
        )
    return _result(
        safe=True,
        reason="ok",
        trajectory_local_xy=adjustment.adjusted_xy,
        obstacle_local_xy=obstacle_local_xy,
        adjustment=adjustment,
        scan_sequence=scan.sequence,
        scan_age_s=scan_age_s,
    )
