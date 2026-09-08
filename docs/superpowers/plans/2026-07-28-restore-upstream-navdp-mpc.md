# Restore Upstream NavDP MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the trajectory-manager/blind-guide layer and restore the fixed-horizon MPC behavior from `InternRobotics/NavDP@bebb436`, while preserving D435 reprojection and the rest of the ROS deployment.

**Architecture:** The planning thread keeps the existing inference and virtual-camera reprojection stages, validates the selected reprojected XY trajectory, and installs that complete trajectory directly into a freshly constructed upstream-semantics MPC. Visualization and diagnostics consume the same installed diffusion trajectory; no persistent path state exists between planning cycles.

**Tech Stack:** Python 3.10, NumPy, CasADi, SciPy, ROS 2, unittest, JSONL diagnostics.

## Global Constraints

- Match `InternRobotics/NavDP@bebb436a9856acbd6ed2a63234a99db6bac2fd3a` for MPC horizon, weights, reference densification, reference selection, and zero-yaw reference states.
- Keep D435 input, calibrated transforms, virtual-camera reprojection, ROS safety, arrival verification, logging, BEV/video recording, and selected-diffusion provenance.
- Use the complete reprojected diffusion trajectory as the MPC source; do not resample it to `N`.
- Remove runtime trajectory history, blind guides, `blind_steps`, dynamic MPC horizon, and trajectory-manager CLI/diagnostic fields.
- Fail closed when the current plan is missing, unsafe, invalid, or cannot construct MPC.
- Preserve unrelated tracked and untracked files.

---

### Task 1: Restore the upstream fixed-horizon controller

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_controllers.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: `Mpc_controller(global_planed_traj, N=15, desired_v=0.5, v_max=0.5, w_max=0.5, ref_gap=3)`.
- Produces: fixed `self.N`, `self.ref_traj_len = N // ref_gap + 1`, `solve(x0)`, `reset()`, and `find_reference_traj(x0, global_planed_traj)`.

- [ ] **Step 1: Replace dynamic-horizon tests with upstream-semantics tests**

Assert that:

```python
controller = Mpc_controller(
    np.array([[0.0, 0.0], [1.0, 0.0]]),
)
self.assertEqual(controller.N, 15)
self.assertEqual(controller.ref_gap, 3)
self.assertEqual(controller.ref_traj_len, 6)
self.assertFalse(hasattr(controller, "blind_steps"))
self.assertFalse(hasattr(controller, "prediction_steps"))
```

Source regression tests must require:

```python
"Q = np.diag([10.0, 10.0, 0.0])"
"R = np.diag([0.02, 0.15])"
"np.zeros((ref_traj.shape[0], 1))"
```

and reject `blind_steps`, `reference_poses_from_xy`, and dynamic `update_ref_traj`.

- [ ] **Step 2: Run the focused tests and verify red**

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_controllers.py' -v
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  RosClientSourceTests.test_controller_cost_tracks_reference_yaw -v
```

Expected: failures report the current dynamic horizon, non-upstream weights,
and tangent-yaw reference behavior.

- [ ] **Step 3: Port the upstream controller**

Use `NavDP-official-bebb436/utils_tasks/tracking_utils.py::MPC_Controller` as
the source of truth, retaining only the local class spelling
`Mpc_controller`. Restore:

```python
N=15
desired_v=0.5
v_max=0.5
w_max=0.5
ref_gap=3
Q = np.diag([10.0, 10.0, 0.0])
R = np.diag([0.02, 0.15])
```

Remove `blind_steps`, `prediction_steps`, `update_ref_traj`,
`reference_poses_from_xy`, and integral-horizon validation. In `solve`, append
zero yaw exactly as upstream:

```python
ref_traj = np.concatenate(
    (ref_traj, np.zeros((ref_traj.shape[0], 1))),
    axis=1,
).reshape(-1, 1)
```

- [ ] **Step 4: Run controller tests and compile**

```bash
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_controllers.py' -v
python -m py_compile scripts/realworld/controllers.py
```

Expected: all controller tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_controllers.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "fix: restore upstream NavDP MPC"
```

### Task 2: Install reprojected diffusion directly

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: finite reprojected diffusion `np.ndarray` with shape `(P, 2)`, `P >= 2`, and `critic_safe=True`.
- Produces: a fresh `Mpc_controller(reprojected_world_xy, desired_v=max_v, v_max=max_v, w_max=max_w)` and immutable `installed_active_traj`.

- [ ] **Step 1: Write source and behavior regressions**

Require the client to:

```python
active_traj = normalize_tracking_trajectory(retained_reprojected_world_xy)
self.mpc = Mpc_controller(
    active_traj,
    desired_v=self.args.max_v,
    v_max=self.args.max_v,
    w_max=self.args.max_w,
)
```

Assert the client source contains no `TrajectoryManager`,
`trajectory_manager.update`, `blind_steps`, or dynamic `N=` argument.

Add `normalize_tracking_trajectory(points)` tests for invalid shape, NaN,
duplicate-only paths, and an unchanged valid diffusion trajectory.

- [ ] **Step 2: Run focused tests and verify red**

```bash
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  RosClientSourceTests GeometryTests -v
```

Expected: source assertions fail while the client still constructs and calls
`TrajectoryManager`.

- [ ] **Step 3: Add direct trajectory validation**

In `wheeled_client_core.py`, replace the trajectory-manager implementation
with:

```python
def normalize_tracking_trajectory(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "trajectory must be finite with shape (N, 2), N >= 2"
        )
    if not np.any(
        np.linalg.norm(points - points[0], axis=1)
        > np.finfo(np.float64).eps
    ):
        raise ValueError("trajectory must contain two distinct points")
    return points
```

This validation must preserve every input point, including consecutive
duplicates, so the upstream index-based interpolation is unchanged. Remove
`TrajectoryUpdate`, `TrajectoryManager`, and manager-only helpers.

- [ ] **Step 4: Rewire the planning thread**

Keep the current reprojection call and `retained_reprojected_world_xy`. When
critic and reprojection are valid, normalize that selected path and install a
new fixed-horizon MPC. When no valid current plan exists, clear
`trajectory_ready`, `installed_active_traj`, MPC, and selected provenance.

Rebuild MPC on every valid plan, as upstream does. Do not pass `N` or
`blind_steps`.

- [ ] **Step 5: Remove manager CLI and diagnostics**

Remove:

```text
--trajectory-point-spacing
--trajectory-join-distance
--trajectory-join-heading-deg
--trajectory-min-remaining
trajectory_reason
trajectory_join_distance_m
trajectory_blind_*
trajectory_diffusion_*
trajectory_projection_advance_m
trajectory_manager_update_ms
mpc_prediction_steps
```

Keep `active_traj`, raw/reprojected geometry, critic, planning time, and
reprojection status/reason.

- [ ] **Step 6: Run client/core tests**

```bash
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_wheeled_client_core.py' -v
python -m py_compile \
  utils_tasks/wheeled_client_core.py \
  scripts/realworld/navdp_imagegoal_client.py
```

Expected: all client/core tests pass.

- [ ] **Step 7: Commit**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "refactor: track reprojected NavDP path directly"
```

### Task 3: Preserve visualization and verify the selective rollback

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py`
- Modify if required: `navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py`
- Modify if required: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`

**Interfaces:**
- Consumes: installed reprojected diffusion trajectory.
- Produces: unchanged D435/BEV/video behavior with no generated blind guides.

- [ ] **Step 1: Update visualization regressions**

Assert that every yellow MPC guide point comes from the installed reprojected
diffusion trajectory and that no chassis anchor or generated blind point is
prepended. Preserve the cyan selected-diffusion overlay and robot marker tests.

- [ ] **Step 2: Run BEV tests and verify red or existing compatibility**

```bash
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_rgb_bev_visualizer.py' -v
```

Expected: tests either fail on old blind-guide expectations or demonstrate
that the renderer already accepts the direct path unchanged.

- [ ] **Step 3: Apply the minimal renderer/documentation adjustment**

Do not change projection math or colors. Update only trajectory-layer wording
or expectations that describe blind/history points.

- [ ] **Step 4: Replay a recent plan**

Read the newest `NavDP-official-bebb436/navdp_logs/*_mpc.jsonl`, select a plan
with valid `reprojected_world_xy`, and assert:

```python
np.testing.assert_array_equal(
    normalize_tracking_trajectory(record["reprojected_world_xy"]),
    installed_path,
)
assert controller.N == 15
assert controller.ref_gap == 3
```

No history from earlier plan records may enter `installed_path`.

- [ ] **Step 5: Run complete verification**

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_controllers.py' -v
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_wheeled_client_core.py' -v
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_rgb_bev_visualizer.py' -v
python -m py_compile \
  scripts/realworld/controllers.py \
  scripts/realworld/navdp_imagegoal_client.py \
  utils_tasks/wheeled_client_core.py \
  utils_tasks/rgb_bev_visualizer.py
git diff --check
```

Expected: all relevant suites and compilation pass.

- [ ] **Step 6: Review and commit**

Verify the final diff does not modify reprojection functions, D435 transform
logic, ROS topics, arrival verification, or video finalization. Then commit
any Task 3 changes:

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/tests/test_rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md
git commit -m "test: verify direct NavDP trajectory rendering"
```
