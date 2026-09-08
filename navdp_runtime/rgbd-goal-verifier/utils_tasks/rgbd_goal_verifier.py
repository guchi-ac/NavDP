from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class VerifierConfig:
    arrival_distance_m: float = 0.85
    distance_scale: float = 1.0
    ratio_test: float = 0.75
    min_matches: int = 8
    min_inliers: int = 6
    min_depth_m: float = 0.1
    max_depth_m: float = 10.0
    pnp_reprojection_error_px: float = 4.0
    pnp_iterations: int = 100
    pnp_confidence: float = 0.99
    required_consecutive: int = 3


@dataclass(frozen=True)
class PoseEstimate:
    success: bool
    distance_m: Optional[float]
    depth_matches: int
    inliers: int
    reason: str


@dataclass(frozen=True)
class ArrivalResult:
    arrived: bool
    candidate: bool
    distance_m: Optional[float]
    goal_features: int
    current_features: int
    raw_matches: int
    good_matches: int
    depth_matches: int
    inliers: int
    consecutive: int
    reason: str


class ConsecutiveArrivalLatch:
    def __init__(self, required_consecutive: int):
        if required_consecutive < 1:
            raise ValueError("required_consecutive must be at least 1")
        self.required_consecutive = required_consecutive
        self.reset()

    def update(self, candidate: bool) -> tuple[bool, int]:
        if self.arrived:
            return True, self.consecutive
        self.consecutive = self.consecutive + 1 if candidate else 0
        self.arrived = self.consecutive >= self.required_consecutive
        return self.arrived, self.consecutive

    def reset(self) -> None:
        self.arrived = False
        self.consecutive = 0


class RgbdGoalVerifier:
    def __init__(self, goal_bgr: np.ndarray, config: VerifierConfig = VerifierConfig()):
        goal_bgr = np.asarray(goal_bgr)
        if goal_bgr.ndim != 3 or goal_bgr.shape[2] != 3:
            raise ValueError("goal_bgr must have shape (H, W, 3)")
        self.config = config
        self._sift = cv2.SIFT_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._goal_keypoints, self._goal_descriptors = self._sift.detectAndCompute(
            cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2GRAY), None
        )
        if self._goal_descriptors is None or len(self._goal_keypoints) < config.min_matches:
            raise ValueError("goal image has no SIFT features")
        self._latch = ConsecutiveArrivalLatch(config.required_consecutive)

    def reset(self) -> None:
        self._latch.reset()

    def update(
        self,
        current_bgr: np.ndarray,
        depth_m: np.ndarray,
        intrinsic: np.ndarray,
    ) -> ArrivalResult:
        current_bgr = np.asarray(current_bgr)
        if current_bgr.ndim != 3 or current_bgr.shape[2] != 3:
            raise ValueError("current_bgr must have shape (H, W, 3)")

        current_keypoints, current_descriptors = self._sift.detectAndCompute(
            cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY), None
        )
        current_features = len(current_keypoints)
        if current_descriptors is None or current_features < self.config.min_matches:
            arrived, consecutive = self._latch.update(False)
            return ArrivalResult(
                arrived,
                False,
                None,
                len(self._goal_keypoints),
                current_features,
                0,
                0,
                0,
                0,
                consecutive,
                "arrived_latched" if arrived else "insufficient_features",
            )

        match_pairs = self._matcher.knnMatch(current_descriptors, self._goal_descriptors, k=2)
        good_matches = [
            pair[0]
            for pair in match_pairs
            if len(pair) == 2 and pair[0].distance < self.config.ratio_test * pair[1].distance
        ]
        if len(good_matches) < self.config.min_matches:
            arrived, consecutive = self._latch.update(False)
            return ArrivalResult(
                arrived,
                False,
                None,
                len(self._goal_keypoints),
                current_features,
                len(match_pairs),
                len(good_matches),
                0,
                0,
                consecutive,
                "arrived_latched" if arrived else "insufficient_matches",
            )

        current_pixels = np.array(
            [current_keypoints[match.queryIdx].pt for match in good_matches],
            dtype=np.float64,
        )
        goal_pixels = np.array(
            [self._goal_keypoints[match.trainIdx].pt for match in good_matches],
            dtype=np.float64,
        )
        pose = estimate_relative_camera_pose(
            current_pixels,
            goal_pixels,
            depth_m,
            intrinsic,
            self.config,
        )
        candidate = bool(
            pose.success
            and pose.distance_m is not None
            and pose.distance_m <= self.config.arrival_distance_m
        )
        arrived, consecutive = self._latch.update(candidate)
        if arrived:
            reason = "arrived" if candidate else "arrived_latched"
        elif candidate:
            reason = "candidate"
        elif pose.success:
            reason = "outside_radius"
        else:
            reason = pose.reason

        return ArrivalResult(
            arrived,
            candidate,
            pose.distance_m,
            len(self._goal_keypoints),
            current_features,
            len(match_pairs),
            len(good_matches),
            pose.depth_matches,
            pose.inliers,
            consecutive,
            reason,
        )


def _sample_depth(depth_m: np.ndarray, u: float, v: float, config: VerifierConfig) -> Optional[float]:
    x = int(round(u))
    y = int(round(v))
    if x < 0 or y < 0 or x >= depth_m.shape[1] or y >= depth_m.shape[0]:
        return None
    window = depth_m[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
    valid = window[
        np.isfinite(window)
        & (window >= config.min_depth_m)
        & (window <= config.max_depth_m)
    ]
    return float(np.median(valid)) if valid.size else None


def estimate_relative_camera_pose(
    current_pixels: np.ndarray,
    goal_pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    config: VerifierConfig,
) -> PoseEstimate:
    current_pixels = np.asarray(current_pixels, dtype=np.float64).reshape(-1, 2)
    goal_pixels = np.asarray(goal_pixels, dtype=np.float64).reshape(-1, 2)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)

    if current_pixels.shape != goal_pixels.shape:
        raise ValueError("current_pixels and goal_pixels must have the same shape")
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D aligned depth image")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")
    if len(current_pixels) < config.min_matches:
        return PoseEstimate(False, None, 0, 0, "insufficient_matches")

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    object_points = []
    image_points = []
    for (u, v), goal_pixel in zip(current_pixels, goal_pixels):
        depth = _sample_depth(depth_m, u, v, config)
        if depth is None:
            continue
        object_points.append(((u - cx) * depth / fx, (v - cy) * depth / fy, depth))
        image_points.append(goal_pixel)

    depth_matches = len(object_points)
    if depth_matches < config.min_matches:
        return PoseEstimate(False, None, depth_matches, 0, "insufficient_depth")

    try:
        success, _, translation, inlier_indices = cv2.solvePnPRansac(
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            intrinsic,
            None,
            iterationsCount=config.pnp_iterations,
            reprojectionError=config.pnp_reprojection_error_px,
            confidence=config.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return PoseEstimate(False, None, depth_matches, 0, "pnp_failed")

    inliers = 0 if inlier_indices is None else int(len(inlier_indices))
    if not success:
        return PoseEstimate(False, None, depth_matches, inliers, "pnp_failed")
    if inliers < config.min_inliers:
        return PoseEstimate(False, None, depth_matches, inliers, "insufficient_inliers")

    return PoseEstimate(
        True,
        float(np.linalg.norm(translation.reshape(3))) * config.distance_scale,
        depth_matches,
        inliers,
        "ok",
    )
