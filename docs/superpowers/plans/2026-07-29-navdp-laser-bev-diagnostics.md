# NavDP Laser BEV Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save raw `/scan` frames in the existing NavDP JSONL diagnostics, associate camera/plan/control records with scan sequence IDs, and overlay the independently generated current laser obstacle map in the MPC BEV MP4 without changing control behavior.

**Architecture:** Add a ROS-independent NumPy laser module that validates raw scan metadata, builds a binary current-hit map, pairs scans to RGB timestamps, and compensates scan-to-camera robot motion through odometry. The ROS client owns the `/scan` subscription and JSONL association; the existing BEV renderer consumes only already-projected obstacle points plus explicit freshness status.

**Tech Stack:** Python 3.10, NumPy, OpenCV, ROS 2 Humble, `sensor_msgs/msg/LaserScan`, unittest, JSONL.

## Global Constraints

- Do not read `/costmap` or `/costmap_local`.
- Do not use laser state in `control_stop_reason`, candidate selection, MPC construction, MPC solve, or `/cmd_vel` publication.
- Default scan topic is `/scan`; expected frame is `laser_frame`.
- Use the audited Mira3 planar laser extrinsic `x=0.042 m`, `y=0`, `yaw=0`.
- Use `0.10 s` RGB/scan synchronization slop and `0.25 s` visualization freshness timeout.
- Use a `0.05 m` map resolution over the existing BEV bounds: forward `6 m`, rear `2 m`, lateral `±4 m`.
- Map semantics are local and explicit: `0=current scan has no hit`, `100=current valid laser hit`.
- Ignore `0.0`, NaN, Inf, below-minimum, and above-maximum ranges.
- Remove scan endpoints inside the Mira3 self rectangle with half-length `0.255 m` and half-width `0.260 m`.
- Keep raw ranges in one `type="scan"` row per accepted scan; associate other rows by `scan_sequence`.
- Render current fresh laser hits in magenta below all trajectory layers.
- Preserve the dirty worktree and do not stage unrelated tracked deletions, modifications, logs, videos, backups, or untracked trees.

---

## File Structure

- Create `navdp_runtime/navdp-imagegoal-client/utils_tasks/laser_obstacle_map.py`: immutable scan/map value objects and all ROS-independent time pairing, geometry, rasterization, and odometry compensation.
- Create `navdp_runtime/navdp-imagegoal-client/tests/test_laser_obstacle_map.py`: direct behavior tests for the new module.
- Modify `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`: make JSON conversion replace non-finite floats with `None`.
- Modify `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`: subscribe to `/scan`, snapshot/log it, pair it to RGB frames, expose CLI arguments, and supply fresh obstacle points to BEV.
- Modify `navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py`: draw laser hits and laser status without owning scan geometry.
- Modify `navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py`: verify laser layer order and stale hiding.
- Modify `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`: verify ROS-client source integration and no control coupling.
- Modify `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`: document scan rows, association fields, map semantics, and BEV colors.

---

### Task 1: ROS-Independent Laser Snapshot and Binary Obstacle Map

**Files:**
- Create: `navdp_runtime/navdp-imagegoal-client/utils_tasks/laser_obstacle_map.py`
- Create: `navdp_runtime/navdp-imagegoal-client/tests/test_laser_obstacle_map.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:268-292`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_laser_obstacle_map.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Produces: `LaserMapConfig(forward_m=6.0, rear_m=2.0, lateral_m=4.0, resolution_m=0.05, laser_x_m=0.042, laser_y_m=0.0, laser_yaw_rad=0.0, self_half_length_m=0.255, self_half_width_m=0.260)`.
- Produces: immutable `LaserScanSnapshot(sequence, stamp_ns, received_at, frame_id, angle_min, angle_increment, range_min, range_max, ranges, odom_xy_yaw)`.
- Produces: immutable `LaserObstacleMap(grid, obstacle_xy, valid_count, invalid_count, self_filtered_count)`.
- Produces: `make_laser_scan_snapshot(...) -> LaserScanSnapshot`.
- Produces: `nearest_scan_snapshot(scans, target_stamp_ns, max_delta_s) -> tuple[Optional[LaserScanSnapshot], Optional[float]]`.
- Produces: `build_current_obstacle_map(snapshot, config) -> LaserObstacleMap`.
- Produces: `laser_points_in_target_base(points_in_source_base, source_odom_xy_yaw, target_odom_xy_yaw) -> np.ndarray`.
- Changes: `_json_value(float("nan"))`, `_json_value(float("inf"))`, and `_json_value(float("-inf"))` return `None`.

- [ ] **Step 1: Write failing snapshot, validation, and time-pairing tests**

Create `tests/test_laser_obstacle_map.py` with:

```python
import math
import unittest

import numpy as np

from utils_tasks.laser_obstacle_map import (
    LaserMapConfig,
    build_current_obstacle_map,
    laser_points_in_target_base,
    make_laser_scan_snapshot,
    nearest_scan_snapshot,
)


def snapshot(
    *,
    sequence=1,
    stamp_ns=1_000_000_000,
    ranges=(1.0,),
    angle_min=0.0,
    angle_increment=0.1,
    range_min=0.1,
    range_max=16.0,
    odom=(0.0, 0.0, 0.0),
):
    return make_laser_scan_snapshot(
        sequence=sequence,
        stamp_ns=stamp_ns,
        received_at=10.0,
        frame_id="laser_frame",
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        ranges=np.asarray(ranges, dtype=np.float32),
        odom_xy_yaw=np.asarray(odom, dtype=np.float64),
    )


class LaserSnapshotTests(unittest.TestCase):
    def test_snapshot_owns_readonly_ranges_and_odom(self):
        ranges = np.array([1.0, 2.0], dtype=np.float32)
        odom = np.array([3.0, 4.0, 0.5])
        result = make_laser_scan_snapshot(
            sequence=7,
            stamp_ns=123,
            received_at=9.0,
            frame_id="laser_frame",
            angle_min=-1.0,
            angle_increment=0.1,
            range_min=0.1,
            range_max=16.0,
            ranges=ranges,
            odom_xy_yaw=odom,
        )
        ranges[:] = 99.0
        odom[:] = 99.0
        np.testing.assert_allclose(result.ranges, [1.0, 2.0])
        np.testing.assert_allclose(result.odom_xy_yaw, [3.0, 4.0, 0.5])
        self.assertFalse(result.ranges.flags.writeable)
        self.assertFalse(result.odom_xy_yaw.flags.writeable)

    def test_rejects_invalid_metadata(self):
        with self.assertRaisesRegex(ValueError, "angle_increment"):
            snapshot(angle_increment=0.0)
        with self.assertRaisesRegex(ValueError, "range bounds"):
            snapshot(range_min=2.0, range_max=1.0)
        with self.assertRaisesRegex(ValueError, "ranges"):
            snapshot(ranges=[])

    def test_pairs_nearest_scan_with_inclusive_slop(self):
        scans = [
            snapshot(sequence=1, stamp_ns=900_000_000),
            snapshot(sequence=2, stamp_ns=1_050_000_000),
        ]
        paired, delta = nearest_scan_snapshot(
            scans,
            target_stamp_ns=1_000_000_000,
            max_delta_s=0.05,
        )
        self.assertEqual(paired.sequence, 2)
        self.assertAlmostEqual(delta, 0.05)

    def test_returns_none_outside_slop(self):
        paired, delta = nearest_scan_snapshot(
            [snapshot(stamp_ns=800_000_000)],
            target_stamp_ns=1_000_000_000,
            max_delta_s=0.10,
        )
        self.assertIsNone(paired)
        self.assertIsNone(delta)
```

- [ ] **Step 2: Write failing obstacle-map and odometry-compensation tests**

Append:

```python
class LaserMapTests(unittest.TestCase):
    def test_projects_hit_with_mira3_laser_offset(self):
        result = build_current_obstacle_map(
            snapshot(ranges=[1.0]),
            LaserMapConfig(),
        )
        np.testing.assert_allclose(result.obstacle_xy, [[1.042, 0.0]], atol=1e-6)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(np.count_nonzero(result.grid == 100), 1)

    def test_ignores_zero_nonfinite_and_out_of_range_values(self):
        result = build_current_obstacle_map(
            snapshot(
                ranges=[0.0, math.nan, math.inf, 0.05, 17.0, 1.0],
                angle_increment=0.2,
            ),
            LaserMapConfig(),
        )
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.invalid_count, 5)

    def test_removes_endpoints_inside_robot_self_rectangle(self):
        result = build_current_obstacle_map(
            snapshot(ranges=[0.15, 1.0], angle_increment=math.pi),
            LaserMapConfig(),
        )
        self.assertEqual(result.self_filtered_count, 1)
        self.assertEqual(result.valid_count, 1)
        np.testing.assert_allclose(result.obstacle_xy, [[-0.958, 0.0]], atol=1e-6)

    def test_places_hit_in_expected_grid_cell(self):
        config = LaserMapConfig(resolution_m=0.05)
        result = build_current_obstacle_map(
            snapshot(ranges=[1.0]),
            config,
        )
        row = round((config.forward_m - 1.042) / config.resolution_m)
        col = round(config.lateral_m / config.resolution_m)
        self.assertEqual(result.grid[row, col], 100)

    def test_compensates_scan_pose_into_target_robot_frame(self):
        converted = laser_points_in_target_base(
            np.array([[1.0, 0.0]]),
            source_odom_xy_yaw=np.array([1.0, 0.0, math.pi / 2]),
            target_odom_xy_yaw=np.array([1.0, 1.0, math.pi / 2]),
        )
        np.testing.assert_allclose(converted, [[0.0, -1.0]], atol=1e-7)
```

- [ ] **Step 3: Write failing standard-JSON test**

Add to the existing JSON writer test class in `tests/test_wheeled_client_core.py`:

```python
def test_json_writer_serializes_nonfinite_scan_ranges_as_null(self):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "scan.jsonl"
        writer = JsonlWriter(path)
        writer.write(
            {
                "type": "scan",
                "ranges": np.array(
                    [0.0, np.nan, np.inf, -np.inf],
                    dtype=np.float32,
                ),
            }
        )
        writer.close()
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(
            json.loads(text)["ranges"],
            [0.0, None, None, None],
        )
```

- [ ] **Step 4: Run tests to verify they fail**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_laser_obstacle_map \
  tests.test_wheeled_client_core -v
```

Expected: import failure for `utils_tasks.laser_obstacle_map` and the JSON
test fails because `json.dumps` emits non-standard `NaN`/`Infinity`.

- [ ] **Step 5: Implement immutable value objects and scan validation**

Create `utils_tasks/laser_obstacle_map.py` with these exact public types and
validation rules:

```python
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

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

    def __post_init__(self):
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
```

`make_laser_scan_snapshot` must copy arrays, set them read-only, require a
positive sequence, non-negative integer stamp, non-empty frame ID, finite
received time/angle/range metadata, nonzero angle increment, and
`0 <= range_min < range_max`.

- [ ] **Step 6: Implement map construction and time/odom helpers**

Implement:

```python
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
```

`build_current_obstacle_map` must:

1. generate angles by `angle_min + arange(len(ranges))*angle_increment`;
2. accept only finite nonzero ranges inside inclusive `[range_min, range_max]`;
3. convert valid endpoints by:

```python
laser_x = ranges * np.cos(angles)
laser_y = ranges * np.sin(angles)
c = np.cos(config.laser_yaw_rad)
s = np.sin(config.laser_yaw_rad)
base_x = config.laser_x_m + c * laser_x - s * laser_y
base_y = config.laser_y_m + s * laser_x + c * laser_y
```

4. discard endpoints satisfying both
   `abs(base_x)<=self_half_length_m` and
   `abs(base_y)<=self_half_width_m`;
5. keep only endpoints inside BEV bounds;
6. create a `uint8` grid with:

```python
height = round((forward_m + rear_m) / resolution_m)
width = round((2.0 * lateral_m) / resolution_m)
row = rint((forward_m - base_x) / resolution_m)
col = rint((lateral_m - base_y) / resolution_m)
```

7. set in-bounds endpoint cells to `100`.

Implement odometry compensation by transforming source-base points to odom,
then odom points to target base using standard planar rotations. When either
odom is `None`, return a copy of the original source-base points.

- [ ] **Step 7: Make JSON conversion standard-compliant**

Before the ndarray branch in `_json_value`, add:

```python
if isinstance(value, (float, np.floating)):
    number = float(value)
    return number if math.isfinite(number) else None
```

Keep integer, ndarray, dict, and list behavior unchanged.

- [ ] **Step 8: Run focused tests and verify they pass**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_laser_obstacle_map \
  tests.test_wheeled_client_core -v
python3 -m py_compile \
  utils_tasks/laser_obstacle_map.py \
  utils_tasks/wheeled_client_core.py
```

Expected: all selected tests pass and both modules compile.

- [ ] **Step 9: Commit Task 1**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/laser_obstacle_map.py \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_laser_obstacle_map.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: build NavDP laser obstacle snapshots"
```

---

### Task 2: ROS `/scan` Subscription, Raw JSONL, and RGB Association

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: Task 1 `LaserMapConfig`, `LaserScanSnapshot`, `make_laser_scan_snapshot`, and `nearest_scan_snapshot`.
- Produces: `FrameSnapshot.stamp_ns`, `FrameSnapshot.laser_snapshot`, and `FrameSnapshot.scan_rgb_dt_s`.
- Produces: `NavdpImageGoalClient._scan_callback(message: LaserScan) -> None`.
- Produces: `type="scan"` JSONL rows plus association fields in plan/control rows.

- [ ] **Step 1: Write failing source-integration tests**

Add a `RosClientSourceTests` test:

```python
def test_client_subscribes_logs_and_pairs_raw_laser_scan_without_control_gate(self):
    source = self.client_source()
    for required in (
        "from sensor_msgs.msg import CameraInfo, Image, LaserScan",
        "self.scan_sub = self.create_subscription(",
        "LaserScan,",
        "args.scan_topic,",
        "self._scan_callback,",
        "qos_profile_sensor_data,",
        "self.recent_scans = deque(maxlen=50)",
        "make_laser_scan_snapshot(",
        "nearest_scan_snapshot(",
        '"type": "scan"',
        '"scan_sequence":',
        '"scan_stamp_ns":',
        '"scan_rgb_dt_s":',
        '"scan_age_s":',
    ):
        self.assertIn(required, source)
    control_source = self.control_loop_source()
    self.assertNotIn("scan_timeout", control_source)
    self.assertNotIn("laser_stale", control_source)
    self.assertNotIn("scan_stale", control_source)
```

Add a CLI defaults test requiring:

```python
for required in (
    'parser.add_argument("--scan-topic", default="/scan")',
    'parser.add_argument("--laser-frame", default="laser_frame")',
    'parser.add_argument("--laser-x", type=float, default=0.042)',
    'parser.add_argument("--laser-y", type=float, default=0.0)',
    'parser.add_argument("--laser-yaw", type=float, default=0.0)',
    'parser.add_argument("--scan-sync-slop", type=float, default=0.10)',
    'parser.add_argument("--scan-timeout", type=float, default=0.25)',
    'parser.add_argument("--laser-map-resolution", type=float, default=0.05)',
):
    self.assertIn(required, source)
```

- [ ] **Step 2: Run source tests and verify they fail**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 tests/test_wheeled_client_core.py \
  RosClientSourceTests.test_client_subscribes_logs_and_pairs_raw_laser_scan_without_control_gate \
  RosClientSourceTests.test_client_exposes_laser_diagnostic_defaults -v
```

Expected: failures list the missing subscription, callback, fields, and CLI
arguments.

- [ ] **Step 3: Add imports, state, subscription, and configuration**

In `navdp_imagegoal_client.py`:

- import `LaserScan`;
- import all Task 1 public laser symbols used below;
- add `stamp_ns`, `laser_snapshot`, and `scan_rgb_dt_s` to `FrameSnapshot`;
- initialize `self.scan_sequence=0`, `self.recent_scans=deque(maxlen=50)`,
  `self.latest_scan=None`, and a throttled frame-error timestamp;
- construct `self.laser_map_config=LaserMapConfig(...)` from CLI arguments;
- add the `/scan` subscription using sensor data QoS.

Validate at client construction:

```python
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
```

- [ ] **Step 4: Implement scan callback and raw JSONL row**

Implement `_scan_callback` so it:

1. rejects a mismatched `message.header.frame_id`;
2. validates the scalar scan metadata before reserving a sequence number;
3. under `data_lock`, increments `scan_sequence` and snapshots latest odom;
4. calls `make_laser_scan_snapshot` with that sequence and odom;
5. under `data_lock`, assigns `self.latest_scan` and appends the snapshot;
6. outside `data_lock`, writes exactly one raw scan record.

Use:

```python
stamp_ns = (
    int(message.header.stamp.sec) * 1_000_000_000
    + int(message.header.stamp.nanosec)
)
record = {
    "type": "scan",
    "wall_time": time.time(),
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
```

Frame mismatch and validation errors must be rate-limited to one log every two
seconds and must not mutate `recent_scans`.

- [ ] **Step 5: Pair scan in the RGB-D callback**

Before constructing `FrameSnapshot`, derive the RGB stamp and select the
nearest scan while holding `data_lock`:

```python
rgb_stamp_ns = (
    int(rgb_message.header.stamp.sec) * 1_000_000_000
    + int(rgb_message.header.stamp.nanosec)
)
laser_snapshot, scan_rgb_dt_s = nearest_scan_snapshot(
    list(self.recent_scans),
    rgb_stamp_ns,
    self.args.scan_sync_slop,
)
```

Store the immutable scan reference, RGB stamp, and signed time delta in the
frame. `None` remains valid and must not drop the RGB-D frame.

- [ ] **Step 6: Add association fields to both plan branches and control rows**

For each accepted/rejected plan record derived from a `FrameSnapshot`, add:

```python
"scan_sequence": (
    None if snapshot.laser_snapshot is None
    else snapshot.laser_snapshot.sequence
),
"scan_stamp_ns": (
    None if snapshot.laser_snapshot is None
    else snapshot.laser_snapshot.stamp_ns
),
"scan_rgb_dt_s": snapshot.scan_rgb_dt_s,
"scan_age_s": (
    None if snapshot.laser_snapshot is None
    else time.monotonic() - snapshot.laser_snapshot.received_at
),
```

In the control loop, copy `latest_scan` and record only:

```python
"scan_sequence": None if latest_scan is None else latest_scan.sequence,
"scan_age_s": (
    None if latest_scan is None
    else cycle_start - latest_scan.received_at
),
```

Do not add scan values to `control_stop_reason`.

- [ ] **Step 7: Add CLI arguments and startup logging**

Add all defaults from Global Constraints. Extend the startup log with:

```text
scan=/scan laser_frame=laser_frame laser_xy_yaw=(0.042,0.000,0.000)
```

Do not change existing defaults for camera, planning, MPC, or control.

- [ ] **Step 8: Run focused tests and compile**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 tests/test_wheeled_client_core.py \
  RosClientSourceTests.test_client_subscribes_logs_and_pairs_raw_laser_scan_without_control_gate \
  RosClientSourceTests.test_client_exposes_laser_diagnostic_defaults \
  RosClientSourceTests.test_client_records_synchronized_mpc_diagnostics -v
python3 -m py_compile scripts/realworld/navdp_imagegoal_client.py
```

Expected: selected source tests pass and the client compiles.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: record synchronized NavDP laser scans"
```

---

### Task 3: Laser Obstacle Layer in MPC RGB BEV

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py`

**Interfaces:**
- Consumes: Task 1 `build_current_obstacle_map` and `laser_points_in_target_base`.
- Extends: `render_mpc_rgb_bev(..., laser_obstacle_xy=None, laser_status="LASER WAITING", laser_age_s=None)`.
- Produces: magenta current-hit rendering below trajectory layers and an explicit laser legend/status.

- [ ] **Step 1: Write failing renderer layer tests**

Add to `RenderingTests`:

```python
def test_draws_magenta_laser_hit_below_yellow_guide(self):
    frame = render_mpc_rgb_bev(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
        np.eye(3),
        np.eye(4),
        current_odom_xy_yaw=np.zeros(3),
        odom_history=np.empty((0, 3)),
        predicted_states=None,
        active_traj=np.array([[1.0, 0.0], [1.1, 0.0]]),
        command=np.zeros(2),
        solve_ms=None,
        odom_status="ODOM OK",
        mpc_status="MPC OK",
        laser_obstacle_xy=np.array([[1.0, 0.0], [1.0, 0.5]]),
        laser_status="LASER OK points=2",
        laser_age_s=0.03,
        config=BevConfig(sample_stride=1),
    )
    np.testing.assert_array_equal(frame[450, 360], [0, 255, 255])
    self.assertTrue(
        np.all(frame[446:455, 311:320] == [255, 0, 255], axis=2).any()
    )

def test_waiting_or_stale_laser_does_not_draw_old_hits(self):
    for status in ("LASER WAITING", "LASER STALE"):
        frame = render_mpc_rgb_bev(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.eye(3),
            np.eye(4),
            current_odom_xy_yaw=np.zeros(3),
            odom_history=np.empty((0, 3)),
            predicted_states=None,
            command=np.zeros(2),
            solve_ms=None,
            odom_status="ODOM OK",
            mpc_status="MPC STALE",
            laser_obstacle_xy=np.array([[1.0, 0.0]]),
            laser_status=status,
            config=BevConfig(sample_stride=1),
        )
        self.assertFalse(
            np.all(frame[446:455, 356:365] == [255, 0, 255], axis=2).any()
        )
```

The first test must also inspect the legend region for a magenta sample and
the ASCII status text through the existing source/overlay helper assertion.

- [ ] **Step 2: Run renderer tests and verify they fail**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_rgb_bev_visualizer.RenderingTests.test_draws_magenta_laser_hit_below_yellow_guide \
  tests.test_rgb_bev_visualizer.RenderingTests.test_waiting_or_stale_laser_does_not_draw_old_hits -v
```

Expected: `render_mpc_rgb_bev` rejects the new keyword arguments.

- [ ] **Step 3: Extend the renderer without adding scan geometry**

Add optional arguments to `render_mpc_rgb_bev`. After RGB-D rasterization and
before the `current_odom_xy_yaw` trajectory block:

```python
if (
    laser_obstacle_xy is not None
    and laser_status.startswith("LASER OK")
):
    laser_pixels = _base_xy_to_pixels(
        np.asarray(laser_obstacle_xy, dtype=np.float64).reshape(-1, 2),
        config,
    )
    for laser_col, laser_row in laser_pixels:
        if (
            0 <= laser_row < config.size_px
            and 0 <= laser_col < config.size_px
        ):
            cv2.circle(
                image,
                (int(laser_col), int(laser_row)),
                2,
                (255, 0, 255),
                -1,
            )
```

Expand the black overlay to `y=140`, keep velocity lines at `24/48/72`, keep
the legend at `96`, add a magenta `laser` legend entry, and place the combined
status at `y=128`. The status must preserve odom/MPC strings and append the
caller-provided laser status plus formatted age when present.

- [ ] **Step 4: Write failing client-to-renderer source test**

Add a source test requiring:

```python
for required in (
    "build_current_obstacle_map(",
    "laser_points_in_target_base(",
    'laser_status = "LASER WAITING"',
    'laser_status = "LASER STALE"',
    'laser_status = "LASER OK points=%d"',
    "laser_obstacle_xy=laser_obstacle_xy",
    "laser_status=laser_status",
    "laser_age_s=laser_age_s",
):
    self.assertIn(required, render_source)
```

Also assert `_render_mpc_bev` never publishes velocity and never mutates
`trajectory_ready`.

- [ ] **Step 5: Build and align only the scan paired to the RGB-D snapshot**

In `_render_mpc_bev`:

1. initialize `laser_obstacle_xy=None`, `laser_age_s=None`, and
   `laser_status="LASER WAITING"`;
2. when `snapshot.laser_snapshot` exists, compute age from `received_at`;
3. if age exceeds `scan_timeout`, set `LASER STALE` and do not build points;
4. otherwise call `build_current_obstacle_map`;
5. compensate points from scan base to RGB-D frame base through the two odom
   poses;
6. set `LASER OK points=<valid_count>`;
7. on a geometry exception, log and use `LASER ERROR` without failing BEV.

Pass points/status/age to `render_mpc_rgb_bev`.

- [ ] **Step 6: Run focused rendering and source tests**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_rgb_bev_visualizer \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_passes_fresh_paired_laser_map_to_mpc_bev -v
python3 -m py_compile \
  utils_tasks/rgb_bev_visualizer.py \
  scripts/realworld/navdp_imagegoal_client.py
```

Expected: all renderer tests and the selected client source test pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: overlay current laser map in NavDP BEV"
```

---

### Task 4: Documentation, Full Verification, and Dry-Run Evidence

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`
- Verify: `navdp_runtime/navdp-imagegoal-client/tests/`
- Produces: fresh dry-run JSONL and MPC BEV MP4 under the existing configured output directories.

**Interfaces:**
- Documents: exact scan JSONL schema, `scan_sequence` association, binary map semantics, magenta BEV layer, freshness states, and diagnostic-only scope.
- Verifies: raw online `/scan` parses under the URDF extrinsic and fresh artifacts are readable.

- [ ] **Step 1: Update runtime documentation**

Add a `雷达诊断` subsection stating:

```text
客户端直接订阅 /scan，不读取 /costmap 或 /costmap_local。每帧原始 LaserScan
以 type=scan 写入同一个 *_mpc.jsonl；plan/control 通过 scan_sequence 关联。
当前二值图只把本帧有效回波终点记为障碍，不表示未知空间或第三方代价。
MPC RGB BEV 中洋红点表示与 RGB 帧时间最近且新鲜的雷达命中。
LASER WAITING/STALE/ERROR 时不绘制旧点。本版本仅记录和显示雷达，不改变
NavDP selected、MPC 或 /cmd_vel。
```

Document all seven CLI arguments and audited extrinsic defaults.

- [ ] **Step 2: Run complete local test suite**

Run:

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=..:${PYTHONPATH} python3 -m unittest \
  test_controllers \
  test_goal_capture \
  test_laser_obstacle_map \
  test_navigator_close.NavigatorCloseClientTests \
  test_rgb_bev_visualizer \
  test_wheeled_client_core -v
```

Expected: all selected tests pass with zero failures and zero errors.

- [ ] **Step 3: Run static verification**

Run:

```bash
cd /home/dev/navdp_deployment
python3 -m py_compile \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/laser_obstacle_map.py \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py
git diff --check
```

Expected: exit code 0. Review `git status --short` and confirm only intended
files from this feature are staged/committed; preserve all unrelated dirty
state.

- [ ] **Step 4: Commit documentation**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md
git commit -m "docs: explain NavDP laser diagnostics"
```

- [ ] **Step 5: Start a bounded dry-run**

Run from the client directory without `--enable-control`:

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
export ROS_DOMAIN_ID=11 ROS2CLI_NO_DAEMON=1 PYTHONUNBUFFERED=1
timeout --signal=INT --kill-after=10s 30s \
  python3 scripts/realworld/navdp_imagegoal_client.py \
    --goal-image goal_tree.jpg
```

Expected: startup says `mode=DRY-RUN`, scan parameters match the audited
defaults, and graceful shutdown finalizes JSONL and both MP4 files. No
nonzero velocity is published by the dry-run client.

- [ ] **Step 6: Verify fresh JSONL associations**

Select the newest JSONL and check it with:

```bash
latest_log=$(
  find /home/dev/NavDP-official-bebb436/navdp_logs \
    -maxdepth 1 -type f -name '*_mpc.jsonl' -printf '%T@ %p\n' |
  sort -nr | head -1 | cut -d' ' -f2-
)
jq -s '{
  scans: (map(select(.type=="scan")) | length),
  plans: (map(select(.type=="plan")) | length),
  controls: (map(select(.type=="control")) | length),
  scan_lengths: (map(select(.type=="scan") | (.ranges|length)) | unique),
  associated_plans:
    (map(select(.type=="plan" and .scan_sequence!=null)) | length),
  associated_controls:
    (map(select(.type=="control" and .scan_sequence!=null)) | length),
  nonstandard_numbers:
    (map(tostring | select(test("NaN|Infinity"))) | length)
}' "$latest_log"
```

Expected: `scans>0`, scan lengths include `1616`, plans/controls exist,
associated counts are nonzero, and `nonstandard_numbers=0`.

- [ ] **Step 7: Verify fresh BEV MP4 visually**

Select the matching newest `*_mpc_rgb_bev.mp4`, extract a contact sheet with
ffmpeg, and inspect it:

```bash
latest_bev=$(
  find /home/dev/NavDP-official-bebb436/navdp_visualizations \
    -maxdepth 1 -type f -name '*_mpc_rgb_bev.mp4' -printf '%T@ %p\n' |
  sort -nr | head -1 | cut -d' ' -f2-
)
diag_dir=$(mktemp -d /tmp/navdp_laser_bev.XXXXXX)
ffmpeg -y -nostdin -loglevel error -i "$latest_bev" \
  -vf "fps=1,scale=480:-2,tile=4x3:padding=4:margin=4" \
  -frames:v 1 "$diag_dir/laser_bev_sheet.jpg"
```

Inspect `laser_bev_sheet.jpg` and verify:

- magenta hits appear in the correct physical direction relative to the live
  RGB-D point cloud;
- status is `LASER OK` on fresh frames;
- selected/guide/MPC paths render above laser hits;
- no stale obstacle points remain when status changes to WAITING/STALE.

- [ ] **Step 8: Record final evidence and scope**

Report exact test counts, newest JSONL/MP4 paths, scan/association counts, and
the visual alignment result. Explicitly state that control behavior remains
unchanged and that obstacle-based trajectory modification is not yet
implemented.
