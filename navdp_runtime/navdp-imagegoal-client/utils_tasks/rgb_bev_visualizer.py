from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BevConfig:
    size_px: int = 720
    lateral_m: float = 4.0
    forward_m: float = 6.0
    rear_m: float = 2.0
    sample_stride: int = 4
    min_depth_m: float = 0.1
    max_depth_m: float = 6.0
    min_height_m: float = -0.3
    max_height_m: float = 2.0
    splat_radius_px: int = 2


def bev_freshness(
    now: float,
    current_odom: Optional[np.ndarray],
    last_odom_time: Optional[float],
    mpc_updated_at: Optional[float],
    odom_timeout: float,
    mpc_timeout: float,
) -> Tuple[str, bool, str]:
    if current_odom is None:
        odom_status = "ODOM WAITING"
    elif last_odom_time is None or now - last_odom_time > odom_timeout:
        odom_status = "ODOM STALE"
    else:
        odom_status = "ODOM OK"

    mpc_fresh = (
        mpc_updated_at is not None
        and now - mpc_updated_at <= mpc_timeout
    )
    return odom_status, mpc_fresh, "MPC OK" if mpc_fresh else "MPC STALE"


def backproject_rgbd_to_base(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    base_from_camera: np.ndarray,
    config: BevConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[
        0 : depth_m.shape[0] : config.sample_stride,
        0 : depth_m.shape[1] : config.sample_stride,
    ]
    depth = depth_m[rows, cols]
    valid = (
        np.isfinite(depth)
        & (depth >= config.min_depth_m)
        & (depth <= config.max_depth_m)
    )
    rows = rows[valid]
    cols = cols[valid]
    depth = depth[valid]

    camera_x = (cols - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    camera_y = (rows - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    camera_points = np.column_stack(
        (camera_x, camera_y, depth, np.ones_like(depth))
    )
    base_points = (base_from_camera @ camera_points.T).T[:, :3]
    keep = (
        np.isfinite(base_points).all(axis=1)
        & (base_points[:, 2] >= config.min_height_m)
        & (base_points[:, 2] <= config.max_height_m)
    )
    return base_points[keep], rgb_bgr[rows, cols][keep], depth[keep]


def world_xy_to_current_base(
    world_xy: np.ndarray,
    current_odom_xy_yaw: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(world_xy, dtype=np.float64) - current_odom_xy_yaw[:2]
    yaw = current_odom_xy_yaw[2]
    rotation = np.array(
        [
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)],
        ]
    )
    return delta @ rotation.T


def _base_xy_to_pixels(base_xy: np.ndarray, config: BevConfig) -> np.ndarray:
    scale = config.size_px / (config.forward_m + config.rear_m)
    col = config.size_px / 2.0 - base_xy[:, 1] * scale
    row = config.forward_m * scale - base_xy[:, 0] * scale
    return np.rint(np.column_stack((col, row))).astype(np.int32)


def _draw_path(
    image: np.ndarray,
    base_xy: Optional[np.ndarray],
    color: Tuple[int, int, int],
    config: BevConfig,
    thickness: int = 4,
) -> None:
    if base_xy is None or len(base_xy) < 2:
        return
    pixels = _base_xy_to_pixels(np.asarray(base_xy)[:, :2], config)
    cv2.polylines(
        image,
        [pixels],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )


def velocity_overlay_lines(
    desired_velocity: np.ndarray,
    actual_velocity: Optional[np.ndarray],
    solve_ms: Optional[float],
) -> Tuple[str, str, str]:
    desired_velocity = np.asarray(desired_velocity, dtype=np.float64).reshape(2)
    desired_line = (
        f"desired: v={desired_velocity[0]:.3f} m/s  "
        f"w={desired_velocity[1]:.3f} rad/s"
    )
    if actual_velocity is None:
        actual_line = "actual:  v=nan m/s  w=nan rad/s"
    else:
        actual_velocity = np.asarray(actual_velocity, dtype=np.float64).reshape(2)
        actual_line = (
            f"actual:  v={actual_velocity[0]:.3f} m/s  "
            f"w={actual_velocity[1]:.3f} rad/s"
        )
    solve_text = "nan" if solve_ms is None else f"{solve_ms:.1f}"
    return desired_line, actual_line, f"solve={solve_text} ms"


def _rasterize_points(
    image: np.ndarray,
    pixels: np.ndarray,
    colors: np.ndarray,
    ranges: np.ndarray,
    radius: int,
) -> None:
    if not len(pixels):
        return
    height, width = image.shape[:2]
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    rows = pixels[:, 1]
    cols = pixels[:, 0]

    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            target_rows = rows + row_offset
            target_cols = cols + col_offset
            visible = (
                (target_rows >= 0)
                & (target_rows < height)
                & (target_cols >= 0)
                & (target_cols < width)
            )
            flat = target_rows[visible] * width + target_cols[visible]
            np.minimum.at(z_buffer, flat, ranges[visible])

    flat_image = image.reshape(-1, 3)
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            target_rows = rows + row_offset
            target_cols = cols + col_offset
            visible = (
                (target_rows >= 0)
                & (target_rows < height)
                & (target_cols >= 0)
                & (target_cols < width)
            )
            flat = target_rows[visible] * width + target_cols[visible]
            sample_ranges = ranges[visible]
            winners = sample_ranges == z_buffer[flat]
            flat_image[flat[winners]] = colors[visible][winners]


def render_mpc_rgb_bev(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    base_from_camera: np.ndarray,
    current_odom_xy_yaw: Optional[np.ndarray],
    odom_history: np.ndarray,
    predicted_states: Optional[np.ndarray],
    command: np.ndarray,
    solve_ms: Optional[float],
    odom_status: str,
    mpc_status: str,
    config: BevConfig,
    actual_velocity: Optional[np.ndarray] = None,
    active_traj: Optional[np.ndarray] = None,
    selected_diffusion: Optional[np.ndarray] = None,
) -> np.ndarray:
    image = np.zeros((config.size_px, config.size_px, 3), dtype=np.uint8)
    centerline_pixels = _base_xy_to_pixels(
        np.array(
            [
                [-config.rear_m, 0.0],
                [config.forward_m, 0.0],
            ],
            dtype=np.float64,
        ),
        config,
    )
    cv2.line(
        image,
        tuple(int(value) for value in centerline_pixels[0]),
        tuple(int(value) for value in centerline_pixels[1]),
        (48, 48, 48),
        1,
        cv2.LINE_8,
    )

    points, colors, ranges = backproject_rgbd_to_base(
        rgb_bgr,
        depth_m,
        intrinsic,
        base_from_camera,
        config,
    )
    visible = (
        (points[:, 0] >= -config.rear_m)
        & (points[:, 0] <= config.forward_m)
        & (np.abs(points[:, 1]) <= config.lateral_m)
    )
    points = points[visible]
    colors = colors[visible]
    ranges = ranges[visible]
    pixels = _base_xy_to_pixels(points[:, :2], config)
    _rasterize_points(
        image,
        pixels,
        colors,
        ranges,
        config.splat_radius_px,
    )

    if current_odom_xy_yaw is not None:
        actual_base = None
        if len(odom_history):
            actual_base = world_xy_to_current_base(
                np.asarray(odom_history)[:, :2],
                current_odom_xy_yaw,
            )
        predicted_base = None
        if predicted_states is not None:
            predicted_base = world_xy_to_current_base(
                np.asarray(predicted_states)[:, :2],
                current_odom_xy_yaw,
            )
        _draw_path(image, actual_base, (0, 255, 0), config)
        _draw_path(image, predicted_base, (0, 0, 255), config)
        if selected_diffusion is not None:
            selected_base = world_xy_to_current_base(
                np.asarray(selected_diffusion)[:, :2],
                current_odom_xy_yaw,
            )
            _draw_path(
                image,
                selected_base,
                (255, 255, 0),
                config,
                thickness=2,
            )
        if active_traj is not None:
            guide_base = world_xy_to_current_base(
                np.asarray(active_traj)[:, :2],
                current_odom_xy_yaw,
            )
            for guide_col, guide_row in _base_xy_to_pixels(
                guide_base,
                config,
            ):
                cv2.circle(
                    image,
                    (int(guide_col), int(guide_row)),
                    3,
                    (0, 255, 255),
                    -1,
                )

    center = _base_xy_to_pixels(np.zeros((1, 2)), config)[0]
    col, row = int(center[0]), int(center[1])
    cv2.rectangle(
        image,
        (col - 10, row - 14),
        (col + 10, row + 14),
        (255, 255, 255),
        2,
    )
    cv2.arrowedLine(
        image,
        (col, row),
        (col, row - 28),
        (255, 255, 255),
        2,
        tipLength=0.3,
    )

    overlay_lines = velocity_overlay_lines(command, actual_velocity, solve_ms)
    cv2.rectangle(image, (0, 0), (650, 112), (0, 0, 0), -1)
    for line, y in zip(overlay_lines, (24, 48, 72)):
        cv2.putText(
            image,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.line(image, (10, 96), (34, 96), (0, 255, 0), 3)
    cv2.putText(
        image,
        "actual",
        (40, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(image, (105, 96), (129, 96), (0, 0, 255), 3)
    cv2.putText(
        image,
        "MPC",
        (135, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(image, (195, 96), (219, 96), (255, 255, 0), 3)
    cv2.putText(
        image,
        "selected",
        (225, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(image, (315, 96), (339, 96), (0, 255, 255), 3)
    cv2.putText(
        image,
        "guide",
        (345, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{odom_status}  {mpc_status}",
        (410, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image
