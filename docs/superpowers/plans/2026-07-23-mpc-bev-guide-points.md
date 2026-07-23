# MPC BEV Guide Points Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the current `active_traj` as yellow discrete guide points in `mpc_rgb_bev.mp4`.

**Architecture:** Copy the active MPC guide trajectory into the immutable control-thread visualization snapshot, expose it to the BEV renderer only while that snapshot is fresh, and render it after the path layers but before the white robot marker. Keep trajectory management and control behavior unchanged.

**Tech Stack:** Python 3, NumPy, OpenCV, `unittest`

## Global Constraints

- Display only `active_traj`; do not add model candidates to this video.
- Keep actual motion green, MPC prediction red, guide points yellow, and the robot marker white.
- Hide guide points whenever the MPC snapshot is stale.
- Do not add runtime parameters or change MPC/control behavior.
- Write and observe failing tests before production changes.

---

### Task 1: Render yellow guide points

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py`

**Interfaces:**
- Consumes: `active_traj: Optional[np.ndarray]` in odometry-frame XY coordinates.
- Produces: `render_mpc_rgb_bev(..., active_traj=None)` with yellow point rendering and a `guide` legend.

- [ ] **Step 1: Write the failing renderer test**

Extend `test_draws_actual_green_mpc_red_and_robot_white` with an active guide
trajectory that does not overlap the existing red and green probes:

```python
active_traj=np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]]),
```

Assert that the guide-point pixel is yellow in BGR ordering:

```python
self.assertGreater(frame[495, 315, 1], 200)
self.assertGreater(frame[495, 315, 2], 200)
self.assertLess(frame[495, 315, 0], 80)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_rgb_bev_visualizer.RenderingTests.test_draws_actual_green_mpc_red_and_robot_white -v
```

Expected: failure because `render_mpc_rgb_bev` does not accept
`active_traj`.

- [ ] **Step 3: Implement the minimal renderer change**

Add an optional trailing parameter:

```python
active_traj: Optional[np.ndarray] = None,
```

When odometry is available, convert the guide trajectory with
`world_xy_to_current_base`. Draw every point as a filled yellow BGR
`(0, 255, 255)` circle of radius 2, with the first point radius 4. Draw the
guide before the white robot marker, and add a yellow `guide` legend entry.

- [ ] **Step 4: Run renderer tests and verify GREEN**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_rgb_bev_visualizer -v
```

Expected: all renderer tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py
git commit -m "feat: render active guide points in MPC BEV"
```

### Task 2: Carry the active trajectory into the video

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: the exact `active_traj` installed into the MPC controller.
- Produces: immutable `MpcVisualizationSnapshot.active_traj` and a renderer call that passes it only for a fresh snapshot.

- [ ] **Step 1: Write failing client wiring tests**

Add source tests that require:

```python
active_traj: np.ndarray
```

on `MpcVisualizationSnapshot`, require a read-only copy named
`active_traj_snapshot` when the control thread saves a solution, and require:

```python
active_traj = mpc_snapshot.active_traj
```

inside the fresh-snapshot branch before passing
`active_traj=active_traj` to `render_mpc_rgb_bev`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.RosClientSourceTests -v
```

Expected: failure because the snapshot and renderer call do not carry
`active_traj`.

- [ ] **Step 3: Implement snapshot and freshness wiring**

Add `active_traj` to `MpcVisualizationSnapshot`. When a solve succeeds, copy
the currently installed `self.mpc.ref_traj`, make the copy read-only, and store
it in the same snapshot as predicted states. In `_render_mpc_bev`, default
`active_traj` to `None`, fill it only under `if mpc_fresh`, and pass it by
keyword to `render_mpc_rgb_bev`.

- [ ] **Step 4: Update the Chinese README**

Document the video layers and exact test commands:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_rgb_bev_visualizer test_wheeled_client_core -v
```

Also document the full relevant-suite command and the pre-existing missing
custom `baselines/navdp/navdp_server.py` limitation.

- [ ] **Step 5: Run all relevant tests and verify GREEN**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_controllers \
  test_goal_capture \
  test_navigator_close.NavigatorCloseClientTests \
  test_rgb_bev_visualizer \
  test_wheeled_client_core -v
```

Expected: all relevant tests pass.

- [ ] **Step 6: Run static verification**

Run:

```bash
python3 -m py_compile \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git diff --check
```

Expected: both commands succeed with no output.

- [ ] **Step 7: Commit Task 2**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: record active guides in MPC BEV video"
```
