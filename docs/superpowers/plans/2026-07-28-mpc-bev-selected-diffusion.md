# MPC BEV Selected Diffusion Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the original selected diffusion associated with the currently installed MPC reference as a cyan line in `mpc_rgb_bev.mp4`.

**Architecture:** Store selected diffusion in the odom frame only after its accepted trajectory is successfully installed into MPC. Copy that installed selection into the same immutable visualization snapshot as the active guide and MPC prediction, then transform and draw all three paths with the RGB-D frame pose.

**Tech Stack:** Python 3, NumPy, OpenCV, `unittest`, ROS2 client source integration.

## Global Constraints

- Cyan `selected` is a 2 px anti-aliased continuous line.
- Yellow `guide` points render over cyan `selected`; the white chassis renders last.
- Rejected or failed-to-install candidates never replace the installed selected diffusion.
- Cyan, yellow, and red odom-frame paths are visible only when MPC and odom are fresh.
- Do not change planning, `TrajectoryManager`, MPC solving, control, or runtime arguments.
- Preserve the user's unrelated `controllers.py` working-tree change.

---

## File Map

- `navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py`
  owns selected-diffusion coordinate conversion, cyan rendering, layer order,
  and the legend.
- `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`
  owns the small immutable installed-selection state transition used by the
  ROS client.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
  owns installed selected state, MPC snapshot synchronization, freshness
  gating, and renderer wiring.
- `navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py`
  owns observable cyan line and layer-order pixel tests.
- `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
  owns accepted/rejected state-transition tests and ROS client integration
  checks.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`
  documents the new layer semantics.

### Task 1: Render the selected diffusion in the BEV

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py:164-218`
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py:101-118`
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py:179-292`

**Interfaces:**
- Consumes: optional odom-frame `selected_diffusion: Optional[np.ndarray]`.
- Produces: `render_mpc_rgb_bev(..., selected_diffusion=None)` with a cyan
  2 px line below yellow guide points and a `selected` legend entry.

- [ ] **Step 1: Write the failing renderer test**

Extend
`test_draws_discrete_guide_points_and_robot_over_the_chassis_anchor` with a
selected path separated from the guide:

```python
selected_diffusion=np.array(
    [[0.50, -0.50], [0.65, -0.50]],
),
```

Assert its center is cyan in BGR order:

```python
self.assertGreater(frame[495, 405, 0], 200)
self.assertGreater(frame[495, 405, 1], 200)
self.assertLess(frame[495, 405, 2], 80)
```

Also add a two-point selected path under one guide segment and assert the
guide's center remains yellow. This catches drawing selected after guide.

- [ ] **Step 2: Run the renderer test and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python -m unittest \
  tests.test_rgb_bev_visualizer.RenderingTests.test_draws_discrete_guide_points_and_robot_over_the_chassis_anchor \
  -v
```

Expected: error because `render_mpc_rgb_bev` does not accept
`selected_diffusion`.

- [ ] **Step 3: Implement the minimal renderer change**

Allow path-specific thickness:

```python
def _draw_path(image, base_xy, color, config, thickness=4):
    ...
    cv2.polylines(
        image,
        [pixels],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )
```

Add the optional renderer parameter:

```python
selected_diffusion: Optional[np.ndarray] = None,
```

Inside the existing odom block, convert and draw selected after predicted
states but before guide points:

```python
if selected_diffusion is not None:
    selected_base = world_xy_to_current_base(
        np.asarray(selected_diffusion)[:, :2],
        current_odom_xy_yaw,
    )
    _draw_path(image, selected_base, (255, 255, 0), config, thickness=2)
```

Add a cyan `selected` sample to the legend and move `guide` and status text
right without exceeding the existing 650 px overlay.

- [ ] **Step 4: Run all renderer tests and verify green**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python -m unittest discover \
  -s tests -p 'test_rgb_bev_visualizer.py' -v
```

Expected: all renderer tests pass.

- [ ] **Step 5: Commit the renderer layer**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py
git commit -m "feat: draw selected diffusion in MPC BEV"
```

### Task 2: Synchronize installed selected diffusion with MPC snapshots

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:24-90`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:1669-1743`
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:15-30`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:60-110`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:168-185`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:672-712`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:917-976`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:1062-1099`

**Interfaces:**
- Produces:
  `update_installed_selected_diffusion(installed, candidate_world_xy, candidate_accepted) -> Optional[np.ndarray]`.
- Produces: `MpcVisualizationSnapshot.selected_diffusion:
  Optional[np.ndarray]`.
- Consumes: Task 1
  `render_mpc_rgb_bev(..., selected_diffusion=selected_diffusion)`.

- [ ] **Step 1: Write failing accepted/rejected state tests**

Add tests using literal arrays:

```python
def test_accepted_candidate_replaces_installed_selected_diffusion(self):
    old = np.array([[0.0, 0.0], [1.0, 0.0]])
    candidate = np.array([[0.5, 0.2], [1.5, 0.4]])

    result = client_core.update_installed_selected_diffusion(
        old,
        candidate,
        candidate_accepted=True,
    )

    np.testing.assert_array_equal(result, candidate)
    self.assertFalse(result.flags.writeable)
    self.assertIsNot(result, candidate)

def test_rejected_candidate_retains_installed_selected_diffusion(self):
    old = np.array([[0.0, 0.0], [1.0, 0.0]])
    rejected = np.array([[0.5, 1.0], [1.5, 1.0]])

    result = client_core.update_installed_selected_diffusion(
        old,
        rejected,
        candidate_accepted=False,
    )

    np.testing.assert_array_equal(result, old)
    self.assertFalse(result.flags.writeable)
    self.assertIsNot(result, old)
```

The production mutation these tests catch is selecting the latest generated
candidate unconditionally instead of the candidate installed in MPC.

- [ ] **Step 2: Run the state tests and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python -m unittest \
  tests.test_wheeled_client_core.GeometryTests.test_accepted_candidate_replaces_installed_selected_diffusion \
  tests.test_wheeled_client_core.GeometryTests.test_rejected_candidate_retains_installed_selected_diffusion \
  -v
```

Expected: errors because `update_installed_selected_diffusion` does not
exist.

- [ ] **Step 3: Implement the immutable state transition**

Add:

```python
def update_installed_selected_diffusion(
    installed,
    candidate_world_xy,
    candidate_accepted: bool,
) -> Optional[np.ndarray]:
    selected = candidate_world_xy if candidate_accepted else installed
    if selected is None:
        return None
    result = np.asarray(selected, dtype=np.float64).copy()
    result.setflags(write=False)
    return result
```

- [ ] **Step 4: Run the state tests and verify green**

Run the Task 2 Step 2 command.

Expected: both tests pass.

- [ ] **Step 5: Write failing client integration checks**

Extend the existing AST-backed MPC visualization tests to require:

```python
selected_diffusion: Optional[np.ndarray]
self.installed_selected_diffusion: Optional[np.ndarray] = None
selected_diffusion=selected_diffusion_snapshot
selected_diffusion = mpc_snapshot.selected_diffusion
selected_diffusion=selected_diffusion
```

Require the fresh-odom block to contain both
`active_traj = mpc_snapshot.active_traj` and
`selected_diffusion = mpc_snapshot.selected_diffusion`. Require the planning
loop to call `update_installed_selected_diffusion` only inside the successful
MPC installation `try` block.

- [ ] **Step 6: Run the integration checks and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python -m unittest \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_snapshots_active_trajectory_for_fresh_bev_rendering \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_hides_odom_frame_paths_when_odom_is_stale_but_mpc_is_fresh \
  -v
```

Expected: failures because the client does not carry selected diffusion.

- [ ] **Step 7: Wire installed selection through the client**

Import `update_installed_selected_diffusion`, add the optional snapshot
field, and initialize:

```python
self.installed_selected_diffusion: Optional[np.ndarray] = None
```

After MPC construction/update and active trajectory installation succeed:

```python
self.installed_selected_diffusion = (
    update_installed_selected_diffusion(
        self.installed_selected_diffusion,
        retained_world_xy,
        trajectory_update.candidate_accepted,
    )
)
```

When `active_traj is None`, clear installed selected under `self.mpc_lock`.
During a successful solve, copy installed selected into a read-only
`selected_diffusion_snapshot`, add it to `MpcVisualizationSnapshot`, gate it
beside the other odom-frame paths in `_render_mpc_bev`, and pass it to
`render_mpc_rgb_bev`.

- [ ] **Step 8: Run focused client tests and verify green**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python -m unittest \
  tests.test_wheeled_client_core.GeometryTests.test_accepted_candidate_replaces_installed_selected_diffusion \
  tests.test_wheeled_client_core.GeometryTests.test_rejected_candidate_retains_installed_selected_diffusion \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_snapshots_active_trajectory_for_fresh_bev_rendering \
  tests.test_wheeled_client_core.RosClientSourceTests.test_client_hides_odom_frame_paths_when_odom_is_stale_but_mpc_is_fresh \
  -v
```

Expected: all focused tests pass.

- [ ] **Step 9: Commit synchronized snapshot wiring**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: snapshot installed selected diffusion"
```

### Task 3: Document and verify the complete feature

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md:212-223`

**Interfaces:**
- Consumes: Task 1 cyan renderer layer and Task 2 synchronized snapshot.
- Produces: user-facing layer documentation and fresh verification evidence.

- [ ] **Step 1: Update the Chinese runtime documentation**

Document the layer order and semantics:

```text
绿色 actual、红色 MPC、青色 selected、黄色 guide、白色底盘
```

State that cyan is the accepted original selected diffusion associated with
the installed MPC reference, while yellow is the complete active trajectory
after trajectory management.

- [ ] **Step 2: Run the complete relevant test suites**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=. python -m unittest discover -s tests -p 'test_rgb_bev_visualizer.py' -v
PYTHONPATH=. python -m unittest discover -s tests -p 'test_wheeled_client_core.py' -v
```

Expected: both suites pass with zero failures.

- [ ] **Step 3: Compile changed Python modules**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
python -m py_compile \
  utils_tasks/rgb_bev_visualizer.py \
  utils_tasks/wheeled_client_core.py \
  scripts/realworld/navdp_imagegoal_client.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Check repository diffs**

Run:

```bash
cd /home/dev/navdp_deployment
git diff --check
git status --short
```

Expected: no whitespace errors; the pre-existing `controllers.py` change and
unrelated untracked files remain untouched.

- [ ] **Step 5: Commit documentation**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md
git commit -m "docs: explain selected diffusion BEV layer"
```
