# Virtual-Camera MPC Reprojection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce NavDP's official `z=-0.2 m` trajectory projection, intersect the resulting real-camera rays with the robot ground plane, and use that reprojected path as the MPC reference while retaining the raw selected diffusion as the cyan diagnostic layer.

**Architecture:** Add two deterministic geometry functions to `wheeled_client_core.py`: one for the official virtual pixel formula and one for real-camera ground-plane reprojection. The planning loop will keep separate raw and reprojected representations, pass only the reprojected world path to `TrajectoryManager`/MPC, and snapshot the raw path only after the corresponding reprojected guide installs successfully.

**Tech Stack:** Python 3, NumPy, ROS2/rclpy, OpenCV visualisation, `unittest`, existing `TrajectoryManager` and CasADi MPC integration.

## Global Constraints

- Work directly in the current `feat/virtual-camera-mpc` checkout; do not create a worktree.
- Preserve the user's existing uncommitted changes in `navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py`; never stage or modify that file for this feature.
- Default `--virtual-camera-height` is exactly `0.2` metres.
- Use the unresized planning-frame intrinsic matrix and RGB image height.
- Cyan remains the raw selected diffusion; yellow remains the guide actually consumed by MPC.
- Reprojection failure retains the previous valid guide, or stops when no guide exists.
- Do not modify the NavDP server, checkpoint, critic selection, MPC weights, occupancy logic, or all-candidate visualisation.
- Use `apply_patch` for every source or documentation edit.

---

### Task 1: Virtual-Camera and Ground-Intersection Geometry

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:691-783`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:24-126`

**Interfaces:**
- Consumes: NavDP local XY waypoints, a `3x3` camera intrinsic matrix, raw image height, `4x4 base_from_camera`, and virtual height.
- Produces:
  - `navdp_virtual_pixels(local_xy: np.ndarray, intrinsic: np.ndarray, image_height: int, virtual_camera_height: float = 0.2) -> np.ndarray`
  - `reproject_navdp_to_ground_base(local_xy: np.ndarray, intrinsic: np.ndarray, image_height: int, base_from_camera: np.ndarray, virtual_camera_height: float = 0.2) -> np.ndarray`
- Raises: `ValueError` with a stable `virtual_reprojection:` prefix for invalid shapes, calibration, rays, intersections, or forward order.

- [ ] **Step 1: Write failing tests for the exact official pixel formula**

Add these tests to `GeometryTests`:

```python
def test_navdp_virtual_pixels_match_official_height_formula(self):
    intrinsic = np.array(
        [[100.0, 0.0, 2.0], [0.0, 120.0, 2.0], [0.0, 0.0, 1.0]]
    )
    local_xy = np.array([[1.0, 0.5], [2.0, -0.5]])

    pixels = client_core.navdp_virtual_pixels(
        local_xy,
        intrinsic,
        image_height=5,
        virtual_camera_height=0.2,
    )

    np.testing.assert_allclose(
        pixels,
        [[-48.0, 26.0], [27.0, 14.0]],
        atol=1e-12,
    )

def test_navdp_virtual_pixels_reject_nonpositive_forward_point(self):
    with self.assertRaisesRegex(
        ValueError,
        "virtual_reprojection: forward distance must be positive",
    ):
        client_core.navdp_virtual_pixels(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.eye(3),
            image_height=5,
        )
```

- [ ] **Step 2: Run the pixel tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.GeometryTests.test_navdp_virtual_pixels_match_official_height_formula \
  test_wheeled_client_core.GeometryTests.test_navdp_virtual_pixels_reject_nonpositive_forward_point \
  -v
```

Expected: both tests error or fail because `navdp_virtual_pixels` does not exist.

- [ ] **Step 3: Implement the minimal official projection helper**

Add to `wheeled_client_core.py` before `trajectory_to_world`:

```python
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
    if (
        not np.isfinite(virtual_camera_height)
        or virtual_camera_height <= 0.0
    ):
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
```

- [ ] **Step 4: Run the pixel tests and verify GREEN**

Run the Step 2 command.

Expected: both tests pass.

- [ ] **Step 5: Write failing tests for level and pitched ground reprojection**

Add:

```python
@staticmethod
def level_optical_transform(height):
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    transform[:3, 3] = [0.0, 0.0, height]
    return transform

def test_virtual_height_equals_real_height_reprojects_identity(self):
    intrinsic = np.array(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )
    local_xy = np.array([[0.5, -0.1], [1.0, 0.2], [2.0, 0.4]])

    ground = client_core.reproject_navdp_to_ground_base(
        local_xy,
        intrinsic,
        image_height=5,
        base_from_camera=self.level_optical_transform(0.2),
        virtual_camera_height=0.2,
    )

    np.testing.assert_allclose(ground, local_xy, atol=1e-12)

def test_pitched_camera_uses_full_optical_rotation(self):
    intrinsic = np.array(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )
    level = self.level_optical_transform(1.0)
    pitch = math.radians(20.0)
    pitch_rotation = np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    pitched = level.copy()
    pitched[:3, :3] = pitch_rotation @ level[:3, :3]
    local_xy = np.array([[1.0, 0.0], [2.0, 0.0]])

    ground = client_core.reproject_navdp_to_ground_base(
        local_xy,
        intrinsic,
        image_height=5,
        base_from_camera=pitched,
        virtual_camera_height=0.2,
    )

    first_ray = pitch_rotation @ np.array([1.0, 0.0, -0.2])
    second_ray = pitch_rotation @ np.array([1.0, 0.0, -0.1])
    expected = np.array(
        [
            [-first_ray[0] / first_ray[2], 0.0],
            [-second_ray[0] / second_ray[2], 0.0],
        ]
    )
    np.testing.assert_allclose(ground, expected, atol=1e-12)
```

- [ ] **Step 6: Run the ground tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.GeometryTests.test_virtual_height_equals_real_height_reprojects_identity \
  test_wheeled_client_core.GeometryTests.test_pitched_camera_uses_full_optical_rotation \
  -v
```

Expected: both tests error because `reproject_navdp_to_ground_base` does not exist.

- [ ] **Step 7: Implement real-camera ray/ground intersection**

Add:

```python
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
```

- [ ] **Step 8: Run the ground tests and verify GREEN**

Run the Step 6 command.

Expected: both tests pass.

- [ ] **Step 9: Write and run invalid-ray regression tests**

Add tests that pass:

- a transform whose optical rays have zero base Z and assert
  `virtual_reprojection: ray is parallel to ground`;
- a camera below `z=0` looking down and assert
  `virtual_reprojection: ground intersection is behind camera`;
- `np.nan` local input and assert `virtual_reprojection: local_xy must be finite`;
- a deliberately ordered trajectory whose reprojected base X decreases and
  assert `virtual_reprojection: trajectory reverses forward progress`.

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_wheeled_client_core.GeometryTests -v
```

Expected: all `GeometryTests` pass.

- [ ] **Step 10: Commit the geometry**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: reproject NavDP virtual camera paths"
```

---

### Task 2: Route the Reprojected Candidate into TrajectoryManager and MPC

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:45-75`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:512-800`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:1064-1780`

**Interfaces:**
- Consumes: `reproject_navdp_to_ground_base(...)` from Task 1.
- Produces:
  - raw `retained_raw_world_xy` for `SelectedDiffusionInstallState` and cyan BEV;
  - `retained_reprojected_world_xy` for `TrajectoryManager` and MPC;
  - a retained active guide when a fresh reprojection is invalid.

- [ ] **Step 1: Write failing source-structure tests for separate raw and control paths**

Add to `RosClientSourceTests`:

```python
def test_client_routes_reprojected_path_to_manager_but_snapshots_raw_selected(self):
    source = self.client_source()

    for required in (
        "reproject_navdp_to_ground_base",
        "raw_selected_world_xy",
        "reprojected_base_xy",
        "reprojected_world_xy",
        "retained_raw_world_xy",
        "retained_reprojected_world_xy",
        "candidate_world_xy=retained_reprojected_world_xy",
        "self.selected_diffusion_state.stage(\n"
        "                                retained_raw_world_xy,",
    ):
        self.assertIn(required, source)
    self.assertNotIn(
        "candidate_world_xy=retained_raw_world_xy",
        source,
    )

def test_invalid_reprojection_advances_manager_without_new_candidate(self):
    source = self.client_source()

    self.assertIn("reprojection_error = None", source)
    self.assertIn("except ValueError as error:", source)
    self.assertIn("reprojection_error = str(error)", source)
    self.assertIn(
        "self.trajectory_manager.update(\n"
        "                        snapshot.odom_xy_yaw[:2]\n"
        "                    )",
        source,
    )
```

- [ ] **Step 2: Run the integration structure tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.RosClientSourceTests.test_client_routes_reprojected_path_to_manager_but_snapshots_raw_selected \
  test_wheeled_client_core.RosClientSourceTests.test_invalid_reprojection_advances_manager_without_new_candidate \
  -v
```

Expected: failures because the client still sends `retained_world_xy`.

- [ ] **Step 3: Import the geometry helper and split planning representations**

Add `reproject_navdp_to_ground_base` to the
`utils_tasks.wheeled_client_core` import list.

Replace the single retained path flow with:

```python
raw_selected_world_xy = trajectory_to_world(
    raw_local_xy,
    snapshot.odom_xy_yaw,
    camera_x=snapshot.camera_xy_yaw[0],
    camera_y=snapshot.camera_xy_yaw[1],
    camera_yaw=snapshot.camera_xy_yaw[2],
)
retained_raw_world_xy = raw_selected_world_xy[
    self.args.skip_trajectory_points :
]

reprojection_error = None
reprojected_base_xy = None
reprojected_world_xy = None
try:
    reprojected_base_xy = reproject_navdp_to_ground_base(
        raw_local_xy,
        snapshot.intrinsic,
        snapshot.rgb_bgr.shape[0],
        snapshot.base_from_camera,
        self.args.virtual_camera_height,
    )
    reprojected_world_xy = trajectory_to_world(
        reprojected_base_xy,
        snapshot.odom_xy_yaw,
    )
    retained_reprojected_world_xy = reprojected_world_xy[
        self.args.skip_trajectory_points :
    ]
except ValueError as error:
    reprojection_error = str(error)
    retained_reprojected_world_xy = None
```

Keep `candidate_world_xy`, the all-sample visualisation, in its existing raw
NavDP geometry.

- [ ] **Step 4: Route valid reprojections and retain history on failures**

Use exactly one of these manager calls per planning cycle:

```python
if retained_reprojected_world_xy is None:
    trajectory_update = self.trajectory_manager.update(
        snapshot.odom_xy_yaw[:2]
    )
else:
    trajectory_update = self.trajectory_manager.update(
        snapshot.odom_xy_yaw[:2],
        candidate_world_xy=retained_reprojected_world_xy,
        candidate_eligible=critic_safe,
    )
```

Set `trajectory_prefix` from `reprojected_world_xy` when reprojection
succeeds, otherwise use an empty prefix. Keep `VisualizationState.trajectory`
equal to `active_traj`, so the existing yellow guide visualises the actual
control reference.

- [ ] **Step 5: Preserve raw selected provenance across MPC installation**

Change the staging argument only:

```python
self.selected_diffusion_state = (
    self.selected_diffusion_state.stage(
        retained_raw_world_xy,
        trajectory_update.candidate_accepted,
    )
)
```

Do not move `commit()` before `Mpc_controller(...)` or
`self.mpc.update_ref_traj(...)`. If reprojection is rejected,
`candidate_accepted` remains false and the last installed cyan trajectory
stays paired with the retained yellow guide.

- [ ] **Step 6: Run the integration structure tests and verify GREEN**

Run the Step 2 command.

Expected: both tests pass.

- [ ] **Step 7: Run trajectory-manager and selected-provenance regressions**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.TrajectoryManagerTests \
  test_wheeled_client_core.GeometryTests.test_pending_selected_survives_failed_install_and_rejected_retry \
  test_wheeled_client_core.GeometryTests.test_unavailable_trajectory_clears_selected_provenance \
  test_wheeled_client_core.RosClientSourceTests.test_client_stages_selected_before_install_and_commits_after_success \
  -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit the MPC routing change**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: feed virtual reprojection to MPC"
```

---

### Task 3: Configuration, Diagnostics, and Operator Documentation

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:755-800`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:1252-1308`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md:60-110`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md:210-270`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:1380-1800`

**Interfaces:**
- Consumes: raw/reprojected arrays and `reprojection_error` from Task 2.
- Produces: `--virtual-camera-height`, JSONL evidence for both geometries, and documented BEV colour semantics.

- [ ] **Step 1: Write failing tests for the CLI and diagnostics schema**

Add to `RosClientSourceTests`:

```python
def test_client_exposes_virtual_camera_height_default(self):
    source = self.client_source()
    client_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "realworld"
        / "navdp_imagegoal_client.py"
    )
    help_result = subprocess.run(
        [sys.executable, str(client_path), "--help"],
        cwd=client_path.parents[2],
        capture_output=True,
        check=False,
        text=True,
    )

    self.assertIn(
        'parser.add_argument("--virtual-camera-height", type=float, default=0.2)',
        source,
    )
    self.assertEqual(help_result.returncode, 0, help_result.stderr)
    self.assertIn("--virtual-camera-height", help_result.stdout)

def test_client_logs_raw_and_reprojected_plan_geometry(self):
    source = self.client_source()

    for required in (
        '"raw_selected_world_xy": retained_raw_world_xy',
        '"reprojected_base_xy": reprojected_base_xy',
        '"reprojected_world_xy": retained_reprojected_world_xy',
        '"virtual_camera_height_m": self.args.virtual_camera_height',
        '"reprojection_status":',
        '"reprojection_reason": reprojection_error',
    ):
        self.assertIn(required, source)
```

- [ ] **Step 2: Run the CLI/diagnostic tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.RosClientSourceTests.test_client_exposes_virtual_camera_height_default \
  test_wheeled_client_core.RosClientSourceTests.test_client_logs_raw_and_reprojected_plan_geometry \
  -v
```

Expected: failures because neither the argument nor fields exist.

- [ ] **Step 3: Add and validate the CLI argument**

Add:

```python
parser.add_argument("--virtual-camera-height", type=float, default=0.2)
```

In `NavdpImageGoalClient.__init__`, reject invalid values before workers
start:

```python
if (
    not np.isfinite(args.virtual_camera_height)
    or args.virtual_camera_height <= 0.0
):
    raise ValueError("--virtual-camera-height must be positive and finite")
```

- [ ] **Step 4: Expand plan diagnostics without hiding rejection**

Replace the ambiguous duplicate fields:

```python
"selected_local_xy": local_xy,
"raw_selected_world_xy": retained_raw_world_xy,
"reprojected_base_xy": reprojected_base_xy,
"reprojected_world_xy": retained_reprojected_world_xy,
"virtual_camera_height_m": self.args.virtual_camera_height,
"reprojection_status": (
    "ok" if reprojection_error is None else "rejected"
),
"reprojection_reason": reprojection_error,
```

Remove the old aliases:

```python
"navdp_world_xy": retained_world_xy,
"candidate_world_xy": retained_world_xy,
```

When reprojection fails, emit a throttled warning containing the stable
reason. Ensure the planning diagnostic is also emitted when no active guide
exists, with `active_traj=None`, so startup rejection is visible in JSONL.

- [ ] **Step 5: Run the CLI/diagnostic tests and verify GREEN**

Run the Step 2 command.

Expected: both tests pass.

- [ ] **Step 6: Update operator documentation and its source test**

Update the trajectory section of
`README_NAVDP_CLIENT_ZH.md` to state:

```text
NavDP 原始 selected diffusion 先按官方虚拟相机高度 0.2 m 投影到像素，
再通过当前 D435 optical TF 与 base_link 地面 z=0 求交。求交后的黄色
active_traj 是 MPC 控制参考；青色轨迹保留未经重投影的原始 selected
diffusion，仅用于对比。
```

Document:

```bash
--virtual-camera-height 0.2
```

Also state that invalid or behind-camera ground intersections reject the new
candidate and retain the previous active guide.

Extend `test_readme_documents_navdp_only_d435_tf_runtime` with these exact
fragments:

```python
"--virtual-camera-height 0.2",
"虚拟相机高度",
"黄色",
"青色",
```

- [ ] **Step 7: Run README and BEV rendering regressions**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.RosClientSourceTests.test_readme_documents_navdp_only_d435_tf_runtime \
  test_rgb_bev_visualizer.RenderingTests.test_draws_cyan_selected_diffusion_under_yellow_guide_points \
  -v
```

Expected: both tests pass. The existing renderer needs no colour change.

- [ ] **Step 8: Commit configuration, diagnostics, and docs**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "docs: expose virtual-camera MPC reprojection"
```

---

### Task 4: Full Verification and Branch Audit

**Files:**
- Verify only; do not edit `controllers.py`.

**Interfaces:**
- Consumes: all deliverables from Tasks 1-3.
- Produces: evidence that geometry, planning integration, rendering, CLI help, and shutdown behaviour still pass.

- [ ] **Step 1: Run static syntax checks**

```bash
python3 -m py_compile \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Run the complete focused test suite**

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_rgb_bev_visualizer \
  test_wheeled_client_core \
  -v
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 3: Verify CLI exposure**

```bash
cd navdp_runtime/navdp-imagegoal-client
python3 scripts/realworld/navdp_imagegoal_client.py --help \
  | rg -- "--virtual-camera-height"
```

Expected: help contains `--virtual-camera-height VIRTUAL_CAMERA_HEIGHT`.

- [ ] **Step 4: Audit the final diff and preserved user change**

```bash
git status --short --branch
git diff feat/trajectory-manager...HEAD -- \
  docs/superpowers/specs/2026-07-28-virtual-camera-mpc-reprojection-design.md \
  docs/superpowers/plans/2026-07-28-virtual-camera-mpc-reprojection.md \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git diff -- \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py
```

Expected:

- feature diff contains only the intended spec, plan, geometry, client,
  README, and test changes;
- `controllers.py` remains modified but unstaged with the user's original
  `Q/R` weight changes;
- no backup, log, video, goal image, or cache file is staged.

- [ ] **Step 5: Record the verified HEAD**

```bash
git log --oneline --decorate -6
git status --short --branch
```

Expected: three implementation commits follow design/plan commits, and only
pre-existing user/untracked workspace files remain outside version control.
