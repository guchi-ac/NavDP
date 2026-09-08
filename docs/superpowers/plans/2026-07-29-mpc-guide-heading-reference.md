# MPC Guide Heading Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the kinematic MPC follow guide tangents so sharp planned turns produce low linear velocity and high angular velocity.

**Architecture:** Convert sampled guide XY points into `[x, y, yaw]` reference poses before each solve. Penalize wrapped yaw error directly in the MPC objective while retaining the existing XY sampling, controller warm start, and atomic `update_ref_traj()` handoff.

**Tech Stack:** Python 3, NumPy, CasADi/Ipopt, `unittest`

## Global Constraints

- Preserve the complete 24-point guide and existing 50x dense interpolation.
- Preserve `update_ref_traj()` and its warm-start arrays.
- Use `N=15`, `desired_v=v_max`, `w_max=0.5 rad/s`.
- Use `Q = diag([10, 10, 5])` and `R = diag([0.02, 0.15])`.
- Preserve camera projection, arrival checks, and command deadman behavior.
- Do not add a rotate-first state machine.

---

### Task 1: Guide Tangent Reference Poses

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_controllers.py`

**Interfaces:**
- Produces: `reference_poses_from_xy(reference_xy, current_yaw) -> np.ndarray`
- Consumes: sampled reference XY from `Mpc_controller.find_reference_traj()`

- [ ] **Step 1: Write failing tangent and angle-wrap tests**

Add tests with hand-derived expectations:

```python
def test_reference_pose_yaw_follows_guide_tangent(self):
    poses = reference_poses_from_xy(
        np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
        current_yaw=0.0,
    )
    np.testing.assert_allclose(poses[:, 2], np.pi / 2.0)

def test_reference_pose_yaw_uses_nearest_equivalent_at_wrap(self):
    angle = np.deg2rad(-179.0)
    points = np.array(
        [[0.0, 0.0], [np.cos(angle), np.sin(angle)]]
    )
    poses = reference_poses_from_xy(
        points,
        current_yaw=np.deg2rad(179.0),
    )
    np.testing.assert_allclose(
        poses[:, 2],
        np.deg2rad(181.0),
        atol=1e-12,
    )

def test_reference_pose_yaw_handles_repeated_points(self):
    poses = reference_poses_from_xy(
        np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        current_yaw=0.3,
    )
    np.testing.assert_allclose(poses[:, 2], 0.0)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=..:${PYTHONPATH} python3 -m unittest \
  test_controllers.UpstreamNavdpMpcTests.test_reference_pose_yaw_follows_guide_tangent \
  test_controllers.UpstreamNavdpMpcTests.test_reference_pose_yaw_uses_nearest_equivalent_at_wrap \
  test_controllers.UpstreamNavdpMpcTests.test_reference_pose_yaw_handles_repeated_points -v
```

Expected: import/error failure because `reference_poses_from_xy` does not exist.

- [ ] **Step 3: Implement tangent pose generation**

In `controllers.py`, add `reference_poses_from_xy()`:

```python
def reference_poses_from_xy(reference_xy, current_yaw):
    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    if (
        reference_xy.ndim != 2
        or reference_xy.shape[1] != 2
        or len(reference_xy) < 2
    ):
        raise ValueError("reference_xy must have shape (N, 2), N >= 2")

    deltas = np.diff(reference_xy, axis=0)
    valid = np.linalg.norm(deltas, axis=1) > np.finfo(np.float64).eps
    if not np.any(valid):
        yaws = np.full(len(reference_xy), current_yaw)
        return np.column_stack((reference_xy, yaws))

    valid_indices = np.flatnonzero(valid)
    segment_yaws = np.unwrap(np.arctan2(deltas[valid, 1], deltas[valid, 0]))
    yaws = np.interp(np.arange(len(reference_xy)), valid_indices, segment_yaws)
    yaws += 2.0 * np.pi * np.round(
        (current_yaw - yaws[0]) / (2.0 * np.pi)
    )
    return np.column_stack((reference_xy, yaws))
```

- [ ] **Step 4: Run the tangent tests and verify GREEN**

Run the Step 2 command. Expected: all three tests pass.

### Task 2: Yaw-Aware MPC and Feasible Reference Speed

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_controllers.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: `reference_poses_from_xy(reference_xy, current_yaw)`
- Produces: unchanged `Mpc_controller.solve(x0) -> (controls, states)`

- [ ] **Step 1: Write failing controller behavior tests**

Add real solver tests:

```python
def test_straight_guide_keeps_angular_velocity_small(self):
    controller = Mpc_controller(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        desired_v=0.15,
        v_max=0.15,
        w_max=0.5,
    )
    controls, _ = controller.solve(np.array([0.0, 0.0, 0.0]))
    self.assertGreater(controls[0, 0], 0.05)
    self.assertLess(abs(controls[0, 1]), 0.02)

def test_right_angle_guide_trades_linear_speed_for_angular_speed(self):
    controller = Mpc_controller(
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        desired_v=0.15,
        v_max=0.15,
        w_max=0.5,
    )
    controls, _ = controller.solve(np.array([0.0, 0.0, 0.0]))
    self.assertLess(controls[0, 0], 0.03)
    self.assertGreater(controls[0, 1], 0.2)
```

Update source-contract assertions to require guide tangent poses,
`Q = np.diag([10.0, 10.0, 5.0])`, and to reject the zero-yaw concatenation.
Update client assertions to require `desired_v=self.args.max_v`.

- [ ] **Step 2: Run controller/client tests and verify RED**

Run:

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=..:${PYTHONPATH} python3 -m unittest \
  test_controllers \
  test_wheeled_client_core.RosClientSourceTests.test_client_installs_complete_reprojected_diffusion_in_upstream_mpc \
  test_wheeled_client_core.RosClientSourceTests.test_controller_cost_matches_upstream_zero_yaw_reference -v
```

Expected: the sharp-turn behavior and yaw-reference assertions fail.

- [ ] **Step 3: Implement wrapped yaw tracking**

In the MPC objective, keep XY cost and add wrapped yaw cost:

```python
Q_xy = np.diag([10.0, 10.0])
Q_yaw = 5.0
R = np.diag([0.02, 0.15])

position_error = (
    opt_states[i, :2] - opt_xs[nn * 3 : nn * 3 + 2].T
)
raw_yaw_error = opt_states[i, 2] - opt_xs[nn * 3 + 2]
yaw_error = ca.atan2(ca.sin(raw_yaw_error), ca.cos(raw_yaw_error))
obj += ca.mtimes([position_error, Q_xy, position_error.T])
obj += Q_yaw * yaw_error**2
```

In `solve()`, replace fixed zero yaw with:

```python
ref_traj = reference_poses_from_xy(ref_traj, x0[2]).reshape(-1, 1)
```

Restore `N=15`; construct the controller with
`desired_v=self.args.max_v`; restore CLI `--max-w` default to `0.50`.

- [ ] **Step 4: Run controller/client tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

### Task 3: Documentation and Full Verification

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`

**Interfaces:**
- Documents: guide tangent yaw references, wrapped yaw cost, feasible reference
  spacing, and the absence of a separate rotate-first mode.

- [ ] **Step 1: Update runtime documentation**

Replace the zero-yaw MPC description with:

```text
MPC 从采样 guide 的局部切线生成参考 yaw，并使用 wrap 后的 yaw 误差。
急转时优化器可降低 v、提高 |w|；系统没有额外的 rotate-first 状态机。
```

Document `N=15`, `desired_v=--max-v`, and `--max-w 0.50`.

- [ ] **Step 2: Run full verification**

Run:

```bash
source /opt/ros/humble/setup.bash
source /home/dev/midea_humanoid_robot/install/setup.bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=..:${PYTHONPATH} python3 -m unittest \
  test_controllers \
  test_goal_capture \
  test_navigator_close.NavigatorCloseClientTests \
  test_rgb_bev_visualizer \
  test_wheeled_client_core -v
```

Expected: all tests pass.

Run:

```bash
python3 -m py_compile \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Review the final diff**

Confirm that the diff contains only tangent reference generation, yaw-aware MPC
cost, feasible deployed parameters, tests, and documentation. Preserve all
unrelated user-owned worktree changes.
