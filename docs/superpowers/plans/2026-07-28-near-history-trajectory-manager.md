# Near-History Trajectory Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the commit-horizon/continuous-overlap manager with a linear-time manager that preserves only the next `1.0 m` of old guide points and lets MPC smooth the concatenated near-history and latest far candidate from the current chassis state.

**Architecture:** Keep one full odom-frame `_active_traj`, but classify its first `history_distance` metres after advancing from the chassis as stable near history; the remainder is the replaceable far candidate. A qualified new candidate is clipped before the same arc boundary, checked only at the seam, and replaces the old far tail. Missing or rejected candidates retain the untraversed old reference.

**Tech Stack:** Python 3, NumPy, unittest, CasADi MPC integration, JSONL diagnostics.

## Global Constraints

- Historical guide points control only the first `1.0 m` of forward path arc by default.
- Do not add a blend zone, weighted point fusion, spline smoothing, commit horizon, or continuous-overlap search.
- The MPC reference must start exactly at the current chassis position.
- Keep candidate validation and seam distance/heading gates.
- Typical 70-point updates must remain linear-time and complete below a generous `0.2 s` regression-test ceiling.
- Preserve unrelated tracked and untracked workspace files.

---

## File Map

- `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`
  owns `TrajectoryManager`, polyline slicing, immutable update results, and update timing.
- `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
  owns manager behavior, performance, client wiring, CLI, and diagnostic source tests.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
  wires the `1.0 m` history parameter into the manager and records the new diagnostics.

### Task 1: Replace overlap matching with bounded near-history replacement

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:14-390`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:83-430`

**Interfaces:**
- Consumes: `TrajectoryManager.update(chassis_xy, candidate_world_xy=None, candidate_eligible=False)`.
- Produces: `TrajectoryManager(..., history_distance=1.0)` and immutable `TrajectoryUpdate` fields `active_traj`, `candidate_accepted`, `reason`, `join_distance_m`, `history_length_m`, `history_point_count`, `far_length_m`, `remaining_length_m`, and `manager_update_ms`.

- [ ] **Step 1: Replace commit/overlap tests with failing near-history tests**

Add focused tests equivalent to:

```python
def test_new_candidate_replaces_only_path_beyond_one_meter(self):
    manager = self.make_manager(history_distance=1.0)
    old = manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [3.0, 0.0]]),
        candidate_eligible=True,
    ).active_traj

    result = manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [1.5, 0.5], [3.0, 1.0]]),
        candidate_eligible=True,
    )

    cumulative = manager._cumulative_lengths(result.active_traj)
    near = cumulative <= 1.0 + 1e-9
    np.testing.assert_allclose(
        result.active_traj[near],
        old[: np.count_nonzero(near)],
        atol=1e-9,
    )
    self.assertTrue(np.any(result.active_traj[:, 1] > 0.5))
    self.assertAlmostEqual(result.history_length_m, 1.0)

def test_old_far_points_enter_near_history_as_chassis_advances(self):
    manager = self.make_manager(history_distance=1.0)
    manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        candidate_eligible=True,
    )

    result = manager.update(
        [0.4, 0.0],
        np.array([[1.4, 0.0], [2.0, 0.5], [3.0, 0.5]]),
        candidate_eligible=True,
    )

    np.testing.assert_array_equal(result.active_traj[0], [0.4, 0.0])
    self.assertTrue(np.any(np.isclose(result.active_traj[:, 0], 1.0, atol=0.03)))
    self.assertLessEqual(result.history_length_m, 1.0 + 1e-9)

def test_candidate_does_not_need_continuous_overlap(self):
    manager = self.make_manager(history_distance=1.0)
    manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [3.0, 0.0]]),
        candidate_eligible=True,
    )

    result = manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [1.4, 0.2], [2.0, 1.0]]),
        candidate_eligible=True,
    )

    self.assertTrue(result.candidate_accepted)
    self.assertEqual(result.reason, "candidate_replaced_far")
```

Retain and adapt existing tests for startup interpolation, chassis re-anchoring,
distance rejection, heading rejection, low critic, malformed input, exhaustion,
copy safety, and constructor validation. Delete tests and constructor arguments
specific to `commit_horizon`, `overlap_length`, and `overlap_distance`.

- [ ] **Step 2: Run the manager tests and verify they fail**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  TrajectoryManagerTests -v
```

Expected: failures because `history_distance` and the new diagnostic fields do
not exist and the old overlap algorithm still rejects non-overlapping updates.

- [ ] **Step 3: Implement exact arc slicing and the linear-time update**

Change `TrajectoryUpdate` to:

```python
@dataclass(frozen=True)
class TrajectoryUpdate:
    active_traj: Optional[np.ndarray]
    candidate_accepted: bool
    reason: str
    join_distance_m: Optional[float]
    history_length_m: float
    history_point_count: int
    far_length_m: float
    remaining_length_m: float
    manager_update_ms: float
```

Change the constructor to accept `history_distance=1.0` and remove the three
commit/overlap arguments. Add a helper that slices a normalized, resampled
polyline at an exact cumulative arc:

```python
@staticmethod
def _point_at_arc(points, cumulative, target):
    index = min(
        int(np.searchsorted(cumulative, target, side="right") - 1),
        len(points) - 2,
    )
    segment_length = cumulative[index + 1] - cumulative[index]
    fraction = (target - cumulative[index]) / segment_length
    return index, points[index] + fraction * (points[index + 1] - points[index])
```

Use that helper to build:

- the old advanced prefix from arc `0` through
  `min(history_distance, old_remaining_length)`;
- the new candidate suffix from arc `history_distance` through its terminal
  point after temporarily prepending the chassis;
- the combined polyline `chassis + old near prefix + new far suffix`.

Perform one nearest projection to advance the old reference, then only
constant-time seam distance/heading checks. If accepted, resample and install
the combined path with reason `candidate_replaced_far`; if rejected, install
the full advanced old reference unchanged. Initialization remains
`chassis + candidate` with reason `initialized`.

Measure `manager_update_ms` with `time.perf_counter()` around the whole update.
Set `history_length_m` to the preserved old prefix arc, `history_point_count`
to the number of output samples at or before that boundary, and `far_length_m`
to `remaining_length_m - history_length_m`.

- [ ] **Step 4: Run the manager tests and verify they pass**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  TrajectoryManagerTests -v
```

Expected: all `TrajectoryManagerTests` pass.

- [ ] **Step 5: Commit the core manager**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "fix: retain only near-field guide history"
```

### Task 2: Wire the history window and new diagnostics into the client

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:150-166`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:735-785`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:1240-1270`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:1436-1480`

**Interfaces:**
- Consumes: the `TrajectoryUpdate` fields and `history_distance` constructor from Task 1.
- Produces: CLI `--trajectory-history-distance 1.0` and JSONL fields `trajectory_history_length_m`, `trajectory_history_point_count`, `trajectory_far_length_m`, and `trajectory_manager_update_ms`.

- [ ] **Step 1: Write failing client source tests**

Update the existing defaults test to require:

```python
self.assertIn(
    '--trajectory-history-distance", type=float, default=1.0',
    source,
)
self.assertIn(
    "history_distance=args.trajectory_history_distance",
    source,
)
self.assertNotIn("--trajectory-commit-horizon", source)
self.assertNotIn("--trajectory-overlap-length", source)
self.assertNotIn("--trajectory-overlap-distance", source)
```

Update the diagnostic test to require:

```python
for required in (
    '"trajectory_history_length_m"',
    "trajectory_update.history_length_m",
    '"trajectory_history_point_count"',
    "trajectory_update.history_point_count",
    '"trajectory_far_length_m"',
    "trajectory_update.far_length_m",
    '"trajectory_manager_update_ms"',
    "trajectory_update.manager_update_ms",
):
    self.assertIn(required, source)
```

Also require the obsolete preserved-length and overlap-error diagnostic names
to be absent.

- [ ] **Step 2: Run the two client tests and verify they fail**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  RosClientSourceTests.test_client_exposes_trajectory_manager_defaults \
  RosClientSourceTests.test_client_records_candidate_decision_and_active_trajectory \
  -v
```

Expected: failures because the client still exposes and records overlap state.

- [ ] **Step 3: Update construction, CLI, JSONL, and informational logging**

Pass:

```python
history_distance=args.trajectory_history_distance
```

Replace the three obsolete arguments with:

```python
parser.add_argument(
    "--trajectory-history-distance",
    type=float,
    default=1.0,
)
```

Replace obsolete diagnostic fields and log placeholders with the four fields
listed in the Task 2 interface. Keep `join_distance_m`,
`remaining_length_m`, candidate decision, reason, and active trajectory.

- [ ] **Step 4: Run the focused client tests and full core test module**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  RosClientSourceTests.test_client_exposes_trajectory_manager_defaults \
  RosClientSourceTests.test_client_records_candidate_decision_and_active_trajectory \
  -v
python -m unittest discover -s tests -p 'test_wheeled_client_core.py' -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the client integration**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: expose near-history trajectory diagnostics"
```

### Task 3: Prove performance and behavior on representative data

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`

**Interfaces:**
- Consumes: the completed manager and diagnostics from Tasks 1 and 2.
- Produces: a regression test that prevents reintroduction of nested overlap scans.

- [ ] **Step 1: Add a failing-or-passing performance regression**

Add:

```python
def test_seventy_point_update_completes_within_realtime_budget(self):
    manager = self.make_manager(history_distance=1.0)
    x = np.linspace(1.0, 4.5, 70)
    manager.update(
        [0.0, 0.0],
        np.column_stack((x, np.zeros_like(x))),
        candidate_eligible=True,
    )

    start = time.perf_counter()
    result = manager.update(
        [0.1, 0.0],
        np.column_stack((x + 0.1, 0.05 * np.sin(x))),
        candidate_eligible=True,
    )
    elapsed = time.perf_counter() - start

    self.assertTrue(result.candidate_accepted)
    self.assertLess(elapsed, 0.2)
    self.assertLess(result.manager_update_ms, 200.0)
```

Import `time` in the test module.

- [ ] **Step 2: Run the performance test repeatedly**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
for run in 1 2 3 4 5; do
  PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
    TrajectoryManagerTests.test_seventy_point_update_completes_within_realtime_budget \
    -v || exit 1
done
```

Expected: five passes, each manager update below `200 ms`.

- [ ] **Step 3: Replay the latest enabled-run candidates through the manager**

Run a read-only inline Python replay over:

```text
NavDP-official-bebb436/navdp_logs/20260728_014504_mpc.jsonl
```

For each plan row containing `snapshot_odom` and `candidate_world_xy`, submit
the candidate to a fresh manager and report accepted/rejected count, maximum
manager update time, whether every active reference begins at the logged
chassis coordinate, and maximum recorded near-history length.

Expected:

- no update exceeds `200 ms`;
- every active trajectory begins at the chassis;
- near-history length never exceeds `1.0 m` apart from floating-point
  tolerance;
- no `join_commit_horizon` or `join_overlap` reason exists.

- [ ] **Step 4: Run complete verification**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
python -m unittest discover -s tests -v
python -m py_compile \
  utils_tasks/wheeled_client_core.py \
  scripts/realworld/navdp_imagegoal_client.py
cd /home/dev/navdp_deployment
git diff --check
```

Expected: all tests pass, compilation succeeds, and `git diff --check` prints
nothing.

- [ ] **Step 5: Commit the performance regression**

If the test was not already committed with Task 1:

```bash
git add navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "test: bound trajectory manager update latency"
```

If there is no uncommitted test change, do not create an empty commit.
