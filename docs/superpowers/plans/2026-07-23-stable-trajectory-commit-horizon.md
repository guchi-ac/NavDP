# Stable Trajectory Commit Horizon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the next `1.0 m` of historical guide trajectory stable, require `0.5 m` of continuous candidate/history overlap before splicing, and align BEV overlays with the RGB-D frame odometry.

**Architecture:** Extend `TrajectoryManager` with an arc-length-aware splice search over resampled candidate points. Select the farthest valid splice after the effective commit horizon, retain the historical prefix, and append only the candidate suffix. Wire three validated runtime parameters and render odometry-frame paths against `FrameSnapshot.odom_xy_yaw`.

**Tech Stack:** Python 3, NumPy, OpenCV, ROS 2 client source, `unittest`

## Global Constraints

- `active_traj[0]` remains the exact current chassis XY.
- Default trajectory point spacing remains `0.05 m`.
- Default commit horizon is `1.0 m`.
- Default overlap window is `0.5 m`.
- Default maximum overlap distance is `0.30 m`.
- A valid overlap window has nondecreasing projection arc length on history.
- Select the valid splice with the greatest history arc length.
- Retain usable history on rejected, invalid, low-critic, or absent candidates.
- Do not change MPC control or optimizer behavior.
- BEV paths use the RGB-D frame's odometry pose; live odometry time still gates freshness.
- Follow test-first RED/GREEN cycles for every production behavior change.

---

### Task 1: Add arc-length commit and overlap splice selection

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: `TrajectoryManager(..., commit_horizon=1.0, overlap_length=0.5, overlap_distance=0.30)`.
- Produces: stable `TrajectoryUpdate.active_traj`, plus `preserved_length_m: float` and `overlap_error_m: Optional[float]`.

- [ ] **Step 1: Add failing constructor and stable-prefix tests**

Update `TrajectoryManagerTests.make_manager` with:

```python
commit_horizon=1.0,
overlap_length=0.5,
overlap_distance=0.30,
```

Add a test that initializes a straight historical path to `3.0 m`, then offers
two candidates that overlap the x-axis beyond `1.0 m` before bending toward
opposite sides. Assert each accepted output keeps every sample with cumulative
arc length `<= 1.0 m` equal to the previous history and starts at the exact
chassis point.

- [ ] **Step 2: Run the stable-prefix test and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.TrajectoryManagerTests.test_commit_horizon_keeps_near_history_stable_across_oscillating_candidates -v
```

Expected: error because the constructor does not accept the new parameters.

- [ ] **Step 3: Add failing continuous-overlap tests**

Add separate tests:

```python
def test_rejects_isolated_crossing_without_continuous_overlap(self):
    manager = self.make_manager()
    manager.update(
        [0.0, 0.0],
        np.array([[0.2, 0.0], [3.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update(
        [0.0, 0.0],
        np.array([[1.2, -1.0], [1.2, 0.0], [1.2, 1.0]]),
        candidate_eligible=True,
    )
    self.assertFalse(result.candidate_accepted)
    self.assertEqual(result.reason, "join_overlap")
```

Add a positive test with candidate points following history from `0.2 m` to
`1.5 m`, then bending away. Assert the candidate is accepted, the splice
preserves at least `1.0 m`, `overlap_error_m <= 0.30`, and the historical
prefix through `preserved_length_m` is unchanged.

- [ ] **Step 4: Add failing short-history extension test**

Initialize a `0.8 m` history with `min_remaining=0.2`, then offer a candidate
that overlaps and extends to `2.0 m`. Assert the candidate is accepted and
`preserved_length_m` is at least `0.6 m` but less than the default `1.0 m`.

- [ ] **Step 5: Run all new manager tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.TrajectoryManagerTests -v
```

Expected: new commit/overlap tests fail while existing tests continue to
describe the previous behavior.

- [ ] **Step 6: Implement cumulative projection helpers**

Add:

```python
@staticmethod
def _cumulative_lengths(points):
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
```

Add `_projection_with_arc(polyline, cumulative, point)` returning:

```python
(distance, segment_index, projection, projection_arc)
```

Retain `_closest_projection` as a compatibility wrapper returning the original
three values.

- [ ] **Step 7: Implement valid-splice search**

Resample the candidate at `point_spacing`. Compute:

```python
history_length = history_cumulative[-1]
effective_commit = min(
    self.commit_horizon,
    max(0.0, history_length - self.min_remaining),
)
```

For candidate indexes that leave at least one following segment:

1. project the candidate point to history and require
   `projection_arc >= effective_commit`;
2. require the splice-point distance to be no greater than `join_distance`;
3. require candidate cumulative length at the index to be at least
   `overlap_length`;
4. select candidate samples covering the preceding `overlap_length`;
5. project those samples to history and require all distances
   `<= overlap_distance`;
6. require projection arcs to be nondecreasing within floating tolerance;
7. apply existing history/connector/candidate-heading checks.

Record every valid splice and select the one with greatest
`projection_arc`, breaking ties with the greatest candidate arc. Combine:

```python
history[: segment_index + 1]
join_point
candidate_resampled[candidate_index:]
```

Snap the first suffix point to `join_point` when the connector is no longer
than `point_spacing`, then normalize and resample.

- [ ] **Step 8: Return diagnostics and rejection reasons**

Extend `TrajectoryUpdate`:

```python
preserved_length_m: float
overlap_error_m: Optional[float]
```

For accepted joins, set both from the selected splice. For initialization or
no accepted splice, use `0.0` and `None`.

When no splice is valid, report the most advanced failed stage:

```text
join_commit_horizon
join_overlap
join_distance
join_heading
```

Retain existing history and all existing behavior for missing, invalid, or
low-critic candidates.

Where an existing focused distance or heading unit test intentionally joins
inside `1.0 m`, override `commit_horizon` and `overlap_length` in that test so
it continues to isolate the original boundary rather than failing at the new
earlier gate.

- [ ] **Step 9: Run manager tests and verify GREEN**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.TrajectoryManagerTests -v
```

Expected: every manager test passes.

- [ ] **Step 10: Commit Task 1**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "fix: stabilize active trajectory splice horizon"
```

### Task 2: Wire parameters, diagnostics, and frame-aligned BEV pose

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`
- Test: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: Task 1 constructor parameters and `TrajectoryUpdate` diagnostics.
- Produces: CLI-configurable commit/overlap rules, JSONL evidence, and RGB-D-frame-aligned BEV overlays.

- [ ] **Step 1: Add failing client source tests**

Require parser defaults:

```text
--trajectory-commit-horizon 1.0
--trajectory-overlap-length 0.5
--trajectory-overlap-distance 0.30
```

Require the corresponding arguments in `TrajectoryManager(...)`, and require
planning diagnostics:

```text
trajectory_preserved_length_m
trajectory_overlap_error_m
```

Add a source/AST test for `_render_mpc_bev` requiring:

```python
frame_odom = (
    None
    if snapshot.odom_xy_yaw is None
    else snapshot.odom_xy_yaw.copy()
)
```

and requiring `frame_odom` as the pose passed to both `bev_freshness` and
`render_mpc_rgb_bev`, rather than `self.latest_odom`.

- [ ] **Step 2: Run RosClientSourceTests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.RosClientSourceTests -v
```

Expected: failures for missing CLI wiring, diagnostics, and frame pose.

- [ ] **Step 3: Wire manager and parser parameters**

Pass:

```python
commit_horizon=args.trajectory_commit_horizon
overlap_length=args.trajectory_overlap_length
overlap_distance=args.trajectory_overlap_distance
```

Add the three parser options with the exact defaults in the global
constraints.

- [ ] **Step 4: Record splice diagnostics**

Add these fields to the planning JSONL record:

```python
"trajectory_preserved_length_m": trajectory_update.preserved_length_m,
"trajectory_overlap_error_m": trajectory_update.overlap_error_m,
```

Include preserved length and overlap error in the concise active-trajectory
log line.

- [ ] **Step 5: Align BEV paths with the RGB-D frame pose**

In `_render_mpc_bev`, stop copying `self.latest_odom` as the render pose.
Create `frame_odom` from `snapshot.odom_xy_yaw`. Use it for:

```python
current_odom=frame_odom
```

in freshness classification and as `current_odom_xy_yaw` in
`render_mpc_rgb_bev`. Keep `last_odom_time`, velocity, history, and MPC
snapshot reads under `data_lock`. Continue hiding odometry-frame paths unless
ODOM and MPC are both fresh.

- [ ] **Step 6: Update README**

Document the three parameters, the stable `1.0 m` prefix, the `0.5 m` /
`0.30 m` overlap rule, rejection behavior, new diagnostics, and the fact that
the yellow path is transformed using the RGB-D frame odometry.

- [ ] **Step 7: Run all relevant tests**

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

- [ ] **Step 8: Run static checks**

Run:

```bash
python3 -m py_compile \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/rgb_bev_visualizer.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git diff --check
```

Expected: both checks succeed with no output.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "fix: gate trajectory updates behind stable history"
```
