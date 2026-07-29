# Official-Camera Control Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make selected-to-guide control geometry use the live D435 color intrinsic matrix with NavDP's fixed level-camera ground plane `Z=-0.2 m`, without using the live D435 mounting transform.

**Architecture:** Add one pure helper that owns the official level optical-to-base transform and its sign convention. The planning loop will use that synthetic transform and odometry-only XY conversion for cyan selected and yellow guide paths, while the RGB-D BEV path will continue receiving the live frame `base_from_camera`. Remove the runtime height override so control geometry is always the official 0.2-metre model.

**Tech Stack:** Python 3, NumPy, ROS2 Humble (`sensor_msgs/CameraInfo`, TF2), OpenCV, `unittest`, AST/source-structure regression tests.

## Global Constraints

- Control geometry uses the live `/cam_head/d435/color/camera_info` `CameraInfo.k`.
- The official camera is level with ground `Z=-0.2 m`, equivalent to synthetic camera origin `z=+0.2 m`.
- Live D435 translation, height, pitch, roll, and yaw must not affect selected-to-guide control geometry.
- Live `base_from_camera` remains in `FrameSnapshot` and remains the RGB-D BEV point-cloud transform.
- Selected and guide odom conversion uses only the planning-frame robot odometry pose.
- Preserve fixed MPC `N=15`, complete diffusion guides, critic behavior, failure gating, and visualization colors.
- Reprojection remains fail closed for invalid points, rays, intersections, and forward progress.
- Do not add a runtime switch between official and live control extrinsics.

---

## File Structure

- `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`
  owns the fixed official height, the synthetic official transform, the
  existing pixel/ray reprojection primitives, and odom conversion helpers.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
  owns ROS data acquisition, planning data flow, MPC installation, diagnostics,
  and the separate live-TF BEV call.
- `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
  owns pure geometry tests and source/AST regression tests for the ROS client.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`
  documents the operator-visible camera geometry and removed CLI option.

### Task 1: Add the fixed official camera transform

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:411-516`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:24-185`

**Interfaces:**
- Consumes: no runtime state.
- Produces:
  - `NAVDP_OFFICIAL_CAMERA_HEIGHT_M: float = 0.2`
  - `navdp_official_base_from_camera() -> np.ndarray`, returning a fresh finite `4x4` level optical-to-base transform with translation `[0.0, 0.0, 0.2]`.

- [ ] **Step 1: Write the failing transform test**

Add this test to `GeometryTests`:

```python
def test_official_camera_transform_is_level_at_fixed_height(self):
    self.assertEqual(client_core.NAVDP_OFFICIAL_CAMERA_HEIGHT_M, 0.2)

    transform = client_core.navdp_official_base_from_camera()

    np.testing.assert_array_equal(
        transform,
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
```

The production change that makes this test pass is the new constant and helper;
the existing test-local `level_optical_transform` does not satisfy this API.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_wheeled_client_core.GeometryTests.test_official_camera_transform_is_level_at_fixed_height \
  -v
```

Expected: `ERROR` because
`wheeled_client_core.navdp_official_base_from_camera` and/or
`NAVDP_OFFICIAL_CAMERA_HEIGHT_M` do not exist.

- [ ] **Step 3: Implement the minimal fixed transform**

Add immediately before `navdp_virtual_pixels`:

```python
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
```

Do not replace or weaken the existing validation inside
`navdp_virtual_pixels` or `reproject_navdp_to_ground_base`.

- [ ] **Step 4: Run focused geometry tests and verify GREEN**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_wheeled_client_core.GeometryTests \
  -v
```

Expected: all geometry tests pass, including level identity, pitched-camera
generic reprojection, and failure cases.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: add official NavDP camera transform"
```

### Task 2: Route control paths through official geometry only

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:55-70`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:622-665`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:708-804`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:850-930`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:1525-1550`

**Interfaces:**
- Consumes:
  - `NAVDP_OFFICIAL_CAMERA_HEIGHT_M`
  - `navdp_official_base_from_camera() -> np.ndarray`
  - existing `reproject_navdp_to_ground_base(...) -> np.ndarray`
  - existing `trajectory_to_world(local_xy, odom_xy_yaw, camera_x=0.0, camera_y=0.0, camera_yaw=0.0) -> np.ndarray`
- Produces:
  - `raw_selected_world_xy` based only on raw selected XY and snapshot odometry.
  - `reprojected_base_xy` based on live intrinsic/image height plus the fixed official transform.
  - `reprojected_world_xy` based only on reprojected XY and snapshot odometry.
  - diagnostics with fixed `"virtual_camera_height_m": NAVDP_OFFICIAL_CAMERA_HEIGHT_M`.
  - unchanged live-TF `base_from_camera=snapshot.base_from_camera` in `_render_mpc_bev`.

- [ ] **Step 1: Write failing source/AST tests for the planning boundary**

Add a helper to `RosClientSourceTests`:

```python
@classmethod
def planning_source(cls):
    source = cls.client_source()
    tree = ast.parse(source)
    planning = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_planning_loop"
    )
    return ast.get_source_segment(source, planning)
```

Add these tests:

```python
def test_control_projection_uses_official_extrinsic_and_live_intrinsic(self):
    planning_source = self.planning_source()

    self.assertIn(
        "official_base_from_camera = navdp_official_base_from_camera()",
        planning_source,
    )
    self.assertIn(
        "algo = navigator_reset(\n"
        "                        snapshot.intrinsic,",
        planning_source,
    )
    self.assertIn("intrinsic=snapshot.intrinsic", planning_source)
    self.assertIn("base_from_camera=official_base_from_camera", planning_source)
    self.assertIn(
        "virtual_camera_height=NAVDP_OFFICIAL_CAMERA_HEIGHT_M",
        planning_source,
    )
    self.assertNotIn(
        "base_from_camera=snapshot.base_from_camera",
        planning_source,
    )

def test_selected_and_guide_world_paths_ignore_live_camera_planar_pose(self):
    planning_source = self.planning_source()

    self.assertIn(
        "raw_selected_world_xy = trajectory_to_world(\n"
        "                    raw_local_xy,\n"
        "                    snapshot.odom_xy_yaw,\n"
        "                )",
        planning_source,
    )
    self.assertIn(
        "reprojected_world_xy = trajectory_to_world(\n"
        "                        reprojected_base_xy,\n"
        "                        snapshot.odom_xy_yaw,\n"
        "                    )",
        planning_source,
    )

def test_live_camera_transform_remains_available_to_rgbd_bev(self):
    source = self.client_source()
    tree = ast.parse(source)
    render_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_render_mpc_bev"
    )
    render_source = ast.get_source_segment(source, render_method)

    self.assertIn(
        "base_from_camera=snapshot.base_from_camera",
        render_source,
    )
```

The production change that makes these tests pass is the planning-call routing;
the helper alone from Task 1 cannot satisfy them.

- [ ] **Step 2: Run the new boundary tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_wheeled_client_core.RosClientSourceTests.test_control_projection_uses_official_extrinsic_and_live_intrinsic \
  tests.test_wheeled_client_core.RosClientSourceTests.test_selected_and_guide_world_paths_ignore_live_camera_planar_pose \
  tests.test_wheeled_client_core.RosClientSourceTests.test_live_camera_transform_remains_available_to_rgbd_bev \
  -v
```

Expected: the first two tests fail because planning still passes
`snapshot.base_from_camera` and `snapshot.camera_xy_yaw`; the BEV retention test
already passes and guards against removing the live transform too broadly.

- [ ] **Step 3: Import and use the official transform in planning**

Extend the imports from `utils_tasks.wheeled_client_core` with:

```python
NAVDP_OFFICIAL_CAMERA_HEIGHT_M,
navdp_official_base_from_camera,
```

Immediately before raw selected world conversion, construct a fresh immutable-by-use
local transform:

```python
official_base_from_camera = navdp_official_base_from_camera()
raw_selected_world_xy = trajectory_to_world(
    raw_local_xy,
    snapshot.odom_xy_yaw,
)
```

Replace the control reprojection call with explicit keywords:

```python
reprojected_base_xy = reproject_navdp_to_ground_base(
    local_xy=raw_local_xy,
    intrinsic=snapshot.intrinsic,
    image_height=snapshot.rgb_bgr.shape[0],
    base_from_camera=official_base_from_camera,
    virtual_camera_height=NAVDP_OFFICIAL_CAMERA_HEIGHT_M,
)
reprojected_world_xy = trajectory_to_world(
    reprojected_base_xy,
    snapshot.odom_xy_yaw,
)
```

Do not change `candidate_world_xy` in this task: it is an all-candidate
diagnostic layer, not the installed selected/guide control geometry.

- [ ] **Step 4: Fix diagnostics to state the fixed official height**

In both successful and failed plan diagnostic dictionaries, replace:

```python
"virtual_camera_height_m": self.args.virtual_camera_height,
```

with:

```python
"virtual_camera_height_m": NAVDP_OFFICIAL_CAMERA_HEIGHT_M,
```

Keep `"camera_pose": snapshot.camera_xy_yaw` as a live-TF diagnostic only.

- [ ] **Step 5: Run focused planning and geometry tests and verify GREEN**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_wheeled_client_core.GeometryTests \
  tests.test_wheeled_client_core.RosClientSourceTests.test_control_projection_uses_official_extrinsic_and_live_intrinsic \
  tests.test_wheeled_client_core.RosClientSourceTests.test_selected_and_guide_world_paths_ignore_live_camera_planar_pose \
  tests.test_wheeled_client_core.RosClientSourceTests.test_live_camera_transform_remains_available_to_rgbd_bev \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_tracks_reprojected_path_directly_but_snapshots_raw_selected \
  tests.test_wheeled_client_core.RosClientSourceTests.test_invalid_reprojection_does_not_retain_an_old_trajectory \
  -v
```

Expected: all listed tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "fix: use official camera extrinsic for MPC guides"
```

### Task 3: Remove the height override and align diagnostics documentation

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:110-126`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:1260-1312`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:885-930`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md:65-85`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md:247-263`

**Interfaces:**
- Consumes: `NAVDP_OFFICIAL_CAMERA_HEIGHT_M = 0.2` from Task 1 and fixed diagnostic usage from Task 2.
- Produces: CLI help without `--virtual-camera-height`, no `args.virtual_camera_height` runtime dependency, and operator documentation that distinguishes fixed control extrinsics from live-TF BEV geometry.

- [ ] **Step 1: Replace the configurable-height test with a failing fixed-height test**

Replace
`test_client_exposes_virtual_camera_height_default` with:

```python
def test_client_fixes_official_camera_height_without_runtime_override(self):
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

    self.assertEqual(help_result.returncode, 0, help_result.stderr)
    self.assertNotIn("--virtual-camera-height", help_result.stdout)
    self.assertNotIn("args.virtual_camera_height", source)
    self.assertIn(
        '"virtual_camera_height_m": NAVDP_OFFICIAL_CAMERA_HEIGHT_M',
        source,
    )
```

Update `test_client_logs_raw_and_reprojected_plan_geometry` so its required
height entry is:

```python
'"virtual_camera_height_m": NAVDP_OFFICIAL_CAMERA_HEIGHT_M',
```

- [ ] **Step 2: Run the fixed-height test and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_fixes_official_camera_height_without_runtime_override \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_logs_raw_and_reprojected_plan_geometry \
  -v
```

Expected: the first test fails because CLI help and constructor validation
still reference `--virtual-camera-height`.

- [ ] **Step 3: Remove the runtime option and constructor validation**

Delete this constructor block:

```python
if (
    not np.isfinite(args.virtual_camera_height)
    or args.virtual_camera_height <= 0.0
):
    raise ValueError(
        "--virtual-camera-height must be positive and finite"
    )
```

Delete this parser argument:

```python
parser.add_argument("--virtual-camera-height", type=float, default=0.2)
```

Run:

```bash
rg -n "args\\.virtual_camera_height|--virtual-camera-height" \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py
```

Expected: no matches.

- [ ] **Step 4: Update the Chinese operator documentation**

Replace the current D435-reprojection paragraph with text that explicitly says:

```markdown
NavDP 原始 selected diffusion 使用真机
`/cam_head/d435/color/camera_info` 发布的彩色相机内参，并按官方 RGB
记录方式固定采用水平相机和相机坐标地面 `Z=-0.2 m`。控制轨迹不使用真机
D435 的安装高度、平移、俯仰、横滚或偏航；黄色 `active_traj` 保留全部
官方几何转换后的 diffusion 点并直接送给 MPC。青色 selected 使用相同的
官方平面方向，仅用于对比。

真机 D435 optical TF 仍用于 RGB-D BEV 点云反投影和 TF 诊断，不参与
selected 到黄色 guide 的控制几何。官方控制高度固定为 `0.2 m`，没有运行时
覆盖参数。
```

In the diagnostics section, replace “虚拟相机高度” with
“固定官方相机高度 `0.2 m`”，and state that `camera_pose` is diagnostic and
does not affect control-path projection.

- [ ] **Step 5: Run the client help and focused tests and verify GREEN**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
if python3 scripts/realworld/navdp_imagegoal_client.py --help \
  | rg -q "virtual-camera-height"; then
  echo "unexpected --virtual-camera-height option" >&2
  exit 1
fi
PYTHONPATH=. python3 -m unittest \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_fixes_official_camera_height_without_runtime_override \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_logs_raw_and_reprojected_plan_geometry \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_has_real_robot_inputs_and_explicit_control_gate \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_captures_full_camera_transform_and_bounded_odom_history \
  -v
```

Expected: help contains no height override and all listed tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "docs: fix NavDP control camera geometry"
```

### Task 4: Run complete verification

**Files:**
- Verify only; no planned production edits.

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: evidence that geometry, MPC, ROS client source structure, BEV rendering, shutdown, and helper clients remain compatible.

- [ ] **Step 1: Run syntax and whitespace checks**

Run:

```bash
python3 -m compileall -q \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py
git diff --check HEAD~3..HEAD
```

Expected: exit status 0 with no output from `git diff --check`.

- [ ] **Step 2: Run the complete tracked client test suite**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python3 -m unittest \
  tests.test_controllers \
  tests.test_goal_capture \
  tests.test_navigator_close.NavigatorCloseClientTests \
  tests.test_rgb_bev_visualizer \
  tests.test_wheeled_client_core \
  -v
```

Expected: all tests pass. Do not run
`NavigatorCloseServerTests`; it requires the environment-specific customized
server file documented in the README.

- [ ] **Step 3: Verify exact dependency boundaries in source**

Run:

```bash
rg -n \
  "NAVDP_OFFICIAL_CAMERA_HEIGHT_M|navdp_official_base_from_camera|snapshot\\.base_from_camera|camera_x=snapshot\\.camera_xy_yaw|args\\.virtual_camera_height" \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py
```

Expected:

- official constant/helper definitions and planning imports are present;
- planning passes `base_from_camera=official_base_from_camera`;
- `_render_mpc_bev` still passes `snapshot.base_from_camera`;
- any remaining `camera_x=snapshot.camera_xy_yaw` belongs only to the
  all-candidate diagnostic conversion, not raw selected or yellow guide;
- no `args.virtual_camera_height`.

- [ ] **Step 4: Inspect final commits and working tree**

Run:

```bash
git log -4 --oneline
git status --short
```

Expected: three focused implementation commits follow the design/plan commits.
Pre-existing unrelated untracked files may remain; no unrelated file is staged
or committed.
