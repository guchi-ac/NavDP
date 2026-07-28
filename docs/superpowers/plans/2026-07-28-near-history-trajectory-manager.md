# Dynamic Blind-Zone Trajectory Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill only the blind path between the chassis and the current diffusion first guide point, preserve every current diffusion guide point in the MPC source path, and extend the MPC decision horizon from diffusion `N` to `K_blind + N` without changing the existing `ref_gap` distance-sampling logic.

**Architecture:** `TrajectoryManager` advances the previous odom-frame path, projects the new candidate first point onto it, densifies only the chassis-to-candidate blind connector, and appends the normalized candidate unchanged. It returns blind/diffusion point counts. `Mpc_controller` keeps public `N` as diffusion steps, builds decision variables for `prediction_steps = blind_steps + N`, and the client rebuilds the controller only when that total dimension changes.

**Tech Stack:** Python 3, NumPy, SciPy interpolation, CasADi/Ipopt, unittest, ROS2 client source integration, JSONL diagnostics.

## Global Constraints

- Do not use a fixed `1.0 m` history boundary, commit horizon, continuous overlap, blend zone, or point averaging.
- Preserve the current diffusion polyline as the complete far-field source sequence after normalization; densify only the blind connector.
- `active_traj[0]` is the current chassis anchor and does not count toward `K_blind` or diffusion `N`.
- `prediction_steps = K_blind + N`.
- Keep `make_ref_denser()`, `desired_v * ref_gap * T`, and `find_reference_traj()` sampling behavior unchanged.
- Keep the current `controllers.py` default `ref_gap` value unchanged.
- Rebuild CasADi only when `prediction_steps` changes; update the reference in place when dimensions match.
- Preserve unrelated tracked and untracked workspace files.

---

## File Map

- `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`
  owns dynamic blind-zone projection, connector densification, persistent
  guide-count metadata, and immutable `TrajectoryUpdate`.
- `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
  owns manager behavior, fallback, performance, client wiring, and diagnostics
  regression tests.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py`
  owns diffusion `N`, blind steps, total CasADi horizon, and unchanged
  reference-distance sampling.
- `navdp_runtime/navdp-imagegoal-client/tests/test_controllers.py`
  owns executable MPC dimension and reference-sampling tests.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
  owns dynamic MPC rebuild/update decisions and plan diagnostics.

### Task 1: Build the dynamic blind-zone reference

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:14-390`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:88-500`

**Interfaces:**
- Consumes: `TrajectoryManager.update(chassis_xy, candidate_world_xy=None, candidate_eligible=False)`.
- Produces: immutable `TrajectoryUpdate(active_traj, candidate_accepted, reason, join_distance_m, blind_length_m, blind_point_count, diffusion_length_m, diffusion_point_count, mpc_prediction_steps, remaining_length_m, manager_update_ms)`.

- [ ] **Step 1: Write failing dynamic-boundary and full-candidate tests**

Add literal behavior tests:

```python
def test_blind_history_ends_at_current_candidate_first_point_projection(self):
    manager = self.make_manager(point_spacing=0.10)
    manager.update(
        [0.0, 0.0],
        np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update(
        [0.0, 0.0],
        np.array([[0.7, 0.0], [1.4, 0.4], [2.0, 0.8]]),
        candidate_eligible=True,
    )
    self.assertTrue(result.candidate_accepted)
    self.assertAlmostEqual(result.blind_length_m, 0.7)
    self.assertEqual(result.diffusion_point_count, 3)
    np.testing.assert_array_equal(
        result.active_traj[-3:],
        [[0.7, 0.0], [1.4, 0.4], [2.0, 0.8]],
    )

def test_blind_history_is_not_capped_at_one_meter(self):
    manager = self.make_manager(point_spacing=0.10)
    manager.update(
        [0.0, 0.0],
        np.array([[0.5, 0.0], [1.0, 0.0], [1.3, 0.0], [2.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update(
        [0.0, 0.0],
        np.array([[1.3, 0.0], [1.7, 0.3], [2.2, 0.6]]),
        candidate_eligible=True,
    )
    self.assertTrue(result.candidate_accepted)
    self.assertAlmostEqual(result.blind_length_m, 1.3)
```

Replace fixed-history tests and remove `history_distance` from the test factory.
Retain startup, chassis advancement, distance/heading rejection, low critic,
malformed input, exhaustion, immutability, and latency coverage.

- [ ] **Step 2: Run focused manager tests and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  TrajectoryManagerTests -v
```

Expected: failures because the current constructor requires
`history_distance`, truncates at `1.0 m`, resamples the diffusion suffix, and
does not return blind/diffusion count metadata.

- [ ] **Step 3: Implement candidate-first projection and blind-only densifying**

Remove `history_distance`, `_suffix_from_arc()`, and fixed-prefix replacement.
Add:

```python
def _densify_preserving_vertices(self, points):
    dense = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        for distance in np.arange(self.point_spacing, length, self.point_spacing):
            dense.append(start + segment * (distance / length))
        dense.append(end)
    return self._normalize_polyline(np.asarray(dense))
```

For a qualified candidate with history:

1. Project `candidate[0]` once onto the advanced history.
2. Form the raw blind path from chassis through the historical projection and
   then to `candidate[0]`.
3. Apply existing seam distance/heading gates. For joins no longer than
   `point_spacing`, ignore the tiny connector direction and compare historical
   and candidate headings directly.
4. Densify only the blind path, drop its chassis and candidate endpoints, and
   concatenate `chassis + blind_guides + candidate`.
5. Set `blind_point_count = len(blind_guides)`,
   `diffusion_point_count = len(candidate)`, and their sum as
   `mpc_prediction_steps`.

On initialization, densify the straight chassis-to-`candidate[0]` connector
only, then append the complete candidate. On rejection/missing candidate,
retain both the advanced old path and the previous count metadata. Reset counts
only when history exhausts.

- [ ] **Step 4: Run manager tests and verify green**

Run the Task 1 Step 2 command. Expected: all `TrajectoryManagerTests` pass.

- [ ] **Step 5: Commit dynamic blind-zone manager**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "fix: fill only the current diffusion blind zone"
```

### Task 2: Extend MPC decisions without changing `ref_gap`

**Files:**
- Modify and add to Git: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py`
- Modify and add to Git: `navdp_runtime/navdp-imagegoal-client/tests/test_controllers.py`

**Interfaces:**
- Consumes: `Mpc_controller(global_planed_traj, N, blind_steps, desired_v, v_max, w_max, ref_gap)`.
- Produces: `N` as diffusion steps, `blind_steps`, `prediction_steps = N + blind_steps`, and `update_ref_traj(global_planed_traj, N=None, blind_steps=None)` that rejects a dimension change.

- [ ] **Step 1: Write failing executable MPC dimension tests**

Add:

```python
def test_blind_steps_extend_diffusion_prediction_horizon(self):
    controller = Mpc_controller(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        N=3,
        blind_steps=2,
        ref_gap=1,
    )
    self.assertEqual(controller.N, 3)
    self.assertEqual(controller.blind_steps, 2)
    self.assertEqual(controller.prediction_steps, 5)
    self.assertEqual(controller.opt_controls.shape, (5, 2))
    self.assertEqual(controller.opt_states.shape, (6, 3))

def test_ref_gap_distance_sampling_is_unchanged_for_extended_horizon(self):
    path = np.column_stack((np.linspace(0.0, 2.0, 201), np.zeros(201)))
    controller = Mpc_controller(
        path,
        N=4,
        blind_steps=2,
        desired_v=0.2,
        ref_gap=2,
    )
    refs = controller.find_reference_traj(np.zeros(3), controller.ref_traj)
    self.assertEqual(len(refs), 4)
    np.testing.assert_allclose(np.diff(refs[:, 0])[:2], 0.04, atol=0.01)
```

Add validation tests for positive integral `N`, nonnegative integral
`blind_steps`, positive integral `ref_gap`, and update rejection when a new
`N + blind_steps` differs from the constructed dimension.

- [ ] **Step 2: Run controller tests and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_controllers.py -v
```

Expected: failures because `blind_steps` and `prediction_steps` do not exist.

- [ ] **Step 3: Implement total prediction dimensions**

Keep the current default `ref_gap` unchanged. Set:

```python
self.N = int(N)
self.blind_steps = int(blind_steps)
self.prediction_steps = self.N + self.blind_steps
```

Use `prediction_steps` for control/state variable shapes, dynamics loops,
objective loops, warm-start arrays, and `ref_traj_len`. Do not change
`make_ref_denser()`, `find_reference_traj()`, or its desired arc formula.

Allow `update_ref_traj(..., N, blind_steps)` only when the new sum equals the
existing `prediction_steps`; update public metadata and the dense path without
rebuilding CasADi.

- [ ] **Step 4: Run controller tests and verify green**

Run the Task 2 Step 2 command. Expected: all controller tests pass.

- [ ] **Step 5: Commit MPC horizon support**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/controllers.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_controllers.py
git commit -m "feat: extend MPC horizon with blind guide steps"
```

### Task 3: Rebuild MPC only for changed total dimensions

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: Task 1 update metadata and Task 2 `Mpc_controller`.
- Produces: client installation branch keyed by `trajectory_update.mpc_prediction_steps` and JSONL blind/diffusion/MPC horizon diagnostics.

- [ ] **Step 1: Write failing client integration tests**

Require the client to:

```python
if (
    self.mpc is None
    or self.mpc.prediction_steps
    != trajectory_update.mpc_prediction_steps
):
    self.mpc = Mpc_controller(
        active_traj,
        N=trajectory_update.diffusion_point_count,
        blind_steps=trajectory_update.blind_point_count,
        ...
    )
else:
    self.mpc.update_ref_traj(
        active_traj,
        N=trajectory_update.diffusion_point_count,
        blind_steps=trajectory_update.blind_point_count,
    )
```

Require CLI help to omit `--trajectory-history-distance`, and diagnostics to
contain `trajectory_blind_length_m`, `trajectory_blind_point_count`,
`trajectory_diffusion_length_m`, `trajectory_diffusion_point_count`, and
`mpc_prediction_steps`.

- [ ] **Step 2: Run focused client tests and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  RosClientSourceTests.test_client_exposes_trajectory_manager_defaults \
  RosClientSourceTests.test_client_rebuilds_mpc_only_for_changed_total_horizon \
  RosClientSourceTests.test_client_records_candidate_decision_and_active_trajectory \
  -v
```

Expected: failures because the client still passes fixed history distance and
always updates one fixed-dimension controller.

- [ ] **Step 3: Update client construction, install branch, logs, and CLI**

Remove `history_distance` construction and CLI. Use Task 3 Step 1’s branch.
Replace obsolete history/far diagnostic names and informational log fields
with blind/diffusion counts, lengths, total prediction steps, and manager time.

- [ ] **Step 4: Run focused and tracked client tests**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
python -m unittest discover -s tests -p 'test_wheeled_client_core.py' -v
python -m unittest discover -s tests -p 'test_rgb_bev_visualizer.py' -v
```

Expected: all tracked client and BEV tests pass.

- [ ] **Step 5: Commit client integration**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: size MPC from blind and diffusion guides"
```

### Task 4: Replay and complete verification

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: evidence for guide preservation, dynamic blind length, MPC horizon metadata, latency, compilation, and tracked tests.

- [ ] **Step 1: Repeat the 70-point latency regression five times**

Run the existing latency test five times. Expected: every manager update below
`200 ms`.

- [ ] **Step 2: Replay the latest enabled log**

Replay `NavDP-official-bebb436/navdp_logs/20260728_014504_mpc.jsonl` through a
fresh manager. Assert every active path starts at the logged chassis, every
accepted update ends with the complete logged candidate, no fixed-history or
overlap reason appears, and manager updates remain below `200 ms`. Report
dynamic blind lengths and `K_blind + N` for each accepted plan.

- [ ] **Step 3: Run fresh complete tracked verification**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
python -m unittest discover -s tests -p 'test_controllers.py'
python -m unittest discover -s tests -p 'test_wheeled_client_core.py'
python -m unittest discover -s tests -p 'test_rgb_bev_visualizer.py'
python -m py_compile \
  utils_tasks/wheeled_client_core.py \
  scripts/realworld/controllers.py \
  scripts/realworld/navdp_imagegoal_client.py
cd /home/dev/navdp_deployment
git diff --check
git status --short --untracked-files=no
```

Expected: all tests pass, compilation succeeds, diff check is empty, and no
tracked modifications remain after commits.

- [ ] **Step 4: Request read-only final review**

Review the implementation against
`docs/superpowers/specs/2026-07-28-near-history-trajectory-manager-design.md`
and fix every Critical or Important finding with a failing regression test.
