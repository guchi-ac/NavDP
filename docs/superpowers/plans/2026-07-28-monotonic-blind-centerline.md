# Monotonic Blind-Zone Centerline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace accumulated chassis-projection history with a persistent planned centerline and generate bounded, forward-monotonic blind guides at the current diffusion spacing.

**Architecture:** `TrajectoryManager` will keep accepted centerline geometry, diffusion suffix, spacing, and progress state separately from its transient chassis-anchored `active_traj`. Chassis projection will only trim progress on the persistent centerline; blind guides will be sampled by increasing centerline arc at the median diffusion interval, and the complete newly accepted diffusion suffix will remain exact.

**Tech Stack:** Python 3, NumPy, unittest, ROS2 client source integration, JSONL replay diagnostics.

## Global Constraints

- Work from the current `feat/virtual-camera-mpc` code, which already contains the trajectory-manager and virtual-camera changes.
- Preserve the existing uncommitted `controllers.py` weight adjustment; do not edit, stage, restore, or commit that file.
- Never store a transient `chassis -> closest projection` connector in persistent trajectory state.
- Define monotonicity by strictly increasing centerline arc, not Euclidean radius from the chassis.
- Use `max(median(diffusion segment lengths), point_spacing)` as the blind sampling interval.
- Keep a newly accepted normalized diffusion candidate exactly unchanged at the `active_traj` tail.
- Keep candidate join-distance and join-heading gates.
- Do not add spline smoothing, blending, moving averages, point fusion, fixed one-metre history, or a commit horizon.
- Do not change the MPC formulation, weights, `ref_gap`, or reference-distance sampling as part of this work.
- Keep `prediction_steps = K_blind + N`.
- Preserve unrelated tracked and untracked workspace files.

---

## File Map

- `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`
  owns persistent centerline state, bounded progress projection, diffusion
  spacing, blind sampling, fallback pruning, and immutable update metadata.
- `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
  owns executable geometry, state, fallback, regression, source-wiring, and
  performance tests.
- `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`
  records the new spacing/progress/turn diagnostics and continues installing
  `K_blind + N` references.
- `NavDP-official-bebb436/navdp_logs/20260728_031428_mpc.jsonl`
  is read-only replay evidence and must not be added to Git.

### Task 1: Separate persistent centerline from transient chassis anchoring

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:14-390`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:288-560`

**Interfaces:**
- Consumes: `TrajectoryManager.update(chassis_xy, candidate_world_xy=None, candidate_eligible=False)`.
- Produces: persistent members `_history_centerline`, `_diffusion_tail`, `_diffusion_start_arc`, `_diffusion_spacing`, and `_last_chassis`; transient `TrajectoryUpdate.active_traj`.

- [ ] **Step 1: Write failing point-density and non-accumulation tests**

Add these tests to `TrajectoryManagerTests`:

```python
def test_blind_guides_use_current_diffusion_median_spacing(self):
    manager = self.make_manager(point_spacing=0.05)
    candidate = np.array(
        [[0.60, 0.0], [0.75, 0.0], [0.90, 0.0], [1.05, 0.0]]
    )

    result = manager.update(
        [0.0, 0.0],
        candidate,
        candidate_eligible=True,
    )

    np.testing.assert_allclose(
        result.active_traj[1:4],
        [[0.15, 0.0], [0.30, 0.0], [0.45, 0.0]],
        atol=1e-9,
    )
    self.assertEqual(result.blind_point_count, 3)
    np.testing.assert_array_equal(result.active_traj[-4:], candidate)


def test_lateral_reanchoring_never_accumulates_projection_corners(self):
    manager = self.make_manager(point_spacing=0.05)
    candidate = np.array(
        [[0.60, 0.0], [0.75, 0.0], [0.90, 0.0], [1.05, 0.0]]
    )
    manager.update([0.0, 0.0], candidate, candidate_eligible=True)

    for index in range(100):
        result = manager.update([0.0, 0.01 if index % 2 else -0.01])

    self.assertLessEqual(result.blind_point_count, 3)
    self.assertLessEqual(len(result.active_traj), 1 + 3 + len(candidate))
    blind_with_seams = np.vstack(
        (
            result.active_traj[: 1 + result.blind_point_count],
            result.active_traj[-len(candidate)],
        )
    )
    headings = np.arctan2(
        np.diff(blind_with_seams[:, 1]),
        np.diff(blind_with_seams[:, 0]),
    )
    turns = np.abs(
        np.arctan2(
            np.sin(np.diff(headings)),
            np.cos(np.diff(headings)),
        )
    )
    self.assertLess(float(np.max(turns, initial=0.0)), math.radians(15.0))
```

Update the old startup expectation from fixed `point_spacing` densification to
candidate-median spacing. Keep the exact-candidate-tail and zero-blind tests.

- [ ] **Step 2: Run the new tests and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  TrajectoryManagerTests.test_blind_guides_use_current_diffusion_median_spacing \
  TrajectoryManagerTests.test_lateral_reanchoring_never_accumulates_projection_corners \
  -v
```

Expected: the spacing test sees approximately `0.05 m` blind spacing, and the
reanchoring test observes accumulated short perpendicular segments.

- [ ] **Step 3: Replace active-history state with explicit persistent state**

In `TrajectoryManager.__init__`, replace `_active_traj` as the source of
history with:

```python
self._history_centerline = None
self._diffusion_tail = None
self._diffusion_start_arc = 0.0
self._diffusion_spacing = self.point_spacing
self._last_chassis = None
```

Keep `_active_traj` only as a read-only-output backing copy if other code needs
it; never use it as input to history advancement.

Add exact helpers:

```python
@staticmethod
def _candidate_spacing(candidate: np.ndarray, minimum: float) -> float:
    lengths = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
    return max(float(np.median(lengths)), minimum)


@classmethod
def _point_at_arc(cls, points: np.ndarray, target_arc: float):
    cumulative = cls._cumulative_lengths(points)
    target = float(np.clip(target_arc, 0.0, cumulative[-1]))
    index = min(
        int(np.searchsorted(cumulative, target, side="right") - 1),
        len(points) - 2,
    )
    segment_length = cumulative[index + 1] - cumulative[index]
    fraction = (target - cumulative[index]) / segment_length
    return index, points[index] + fraction * (
        points[index + 1] - points[index]
    )


@classmethod
def _trim_from_arc(cls, points: np.ndarray, start_arc: float) -> np.ndarray:
    index, boundary = cls._point_at_arc(points, start_arc)
    return cls._normalize_polyline(
        np.vstack((boundary, points[index + 1 :]))
    )


@classmethod
def _prefix_through_arc(cls, points: np.ndarray, end_arc: float) -> np.ndarray:
    index, boundary = cls._point_at_arc(points, end_arc)
    return cls._normalize_polyline(
        np.vstack((points[: index + 1], boundary))
    )


@classmethod
def _sample_at_arcs(cls, points: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    if len(arcs) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [cls._point_at_arc(points, float(arc))[1] for arc in arcs],
        dtype=np.float64,
    )
```

Handle a zero-length prefix without calling `_normalize_polyline`; return a
single boundary point internally, while keeping all externally returned active
trajectories valid.

- [ ] **Step 4: Implement centerline initialization and transient sampling**

For initialization:

```python
spacing = self._candidate_spacing(candidate, self.point_spacing)
connector_length = float(np.linalg.norm(candidate[0] - chassis))
centerline = self._normalize_polyline(np.vstack((chassis, candidate)))
diffusion_start_arc = connector_length
blind_arcs = np.arange(spacing, diffusion_start_arc, spacing)
blind_guides = self._sample_at_arcs(centerline, blind_arcs)
active = self._assemble_active_trajectory(chassis, blind_guides, candidate)
```

Commit the persistent state only after the proposed active length satisfies
`min_remaining`. Store `candidate.copy()` as `_diffusion_tail`.

For active reconstruction, always calculate:

```python
blind_arcs = np.arange(
    self._diffusion_spacing,
    self._diffusion_start_arc,
    self._diffusion_spacing,
)
blind_guides = self._sample_at_arcs(
    self._history_centerline,
    blind_arcs,
)
active = self._assemble_active_trajectory(
    chassis,
    blind_guides,
    self._diffusion_tail,
)
```

Do not call `_densify_preserving_vertices` or `_build_blind_path`. Remove those
helpers after all tests stop referencing them.

- [ ] **Step 5: Implement candidate replacement without a chassis anchor**

After history advancement, use the existing projection calculation on
`_history_centerline`. For equal-distance projections, update
`_projection_with_arc` so an existing earlier best is retained within
`1e-9 m`.

Build accepted persistent geometry as:

```python
history_prefix = self._prefix_through_arc(
    self._history_centerline,
    join_arc,
)
proposed_centerline = self._normalize_polyline(
    np.vstack((history_prefix, candidate))
)
proposed_start_arc = (
    self._polyline_length(history_prefix)
    + float(np.linalg.norm(candidate[0] - history_prefix[-1]))
)
proposed_spacing = self._candidate_spacing(
    candidate,
    self.point_spacing,
)
```

Run the existing join-distance and join-heading gates before constructing this
state. Build a transient proposed active trajectory from the proposed values,
validate `min_remaining` and the first-distinct-reference forward check, then
atomically replace all persistent members.

Never insert `chassis` into `proposed_centerline`.

- [ ] **Step 6: Run all manager tests and make them green**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  TrajectoryManagerTests -v
```

Expected: all manager tests pass. Adapt assertions that assumed fixed
`point_spacing` density, but do not weaken exact diffusion-tail, join-gate,
immutability, exhaustion, or latency assertions.

- [ ] **Step 7: Commit state separation**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "fix: separate blind centerline from chassis anchor"
```

Before committing, run `git diff --cached --name-only` and verify
`scripts/realworld/controllers.py` is absent.

### Task 2: Enforce bounded forward progress and fallback pruning

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py:40-390`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:288-620`

**Interfaces:**
- Consumes: Task 1 persistent centerline state.
- Produces: `_advance_centerline(chassis) -> tuple[bool, float]`, strictly forward fallback references, and remaining exact diffusion suffix metadata.

- [ ] **Step 1: Write failing bounded-progress and fallback tests**

Add:

```python
def test_chassis_projection_cannot_jump_across_self_crossing_centerline(self):
    manager = self.make_manager(point_spacing=0.05)
    candidate = np.array(
        [
            [0.30, 0.0],
            [0.60, 0.30],
            [0.30, 0.60],
            [0.0, 0.30],
            [0.30, 0.0],
            [0.60, -0.30],
        ]
    )
    manager.update([0.0, 0.0], candidate, candidate_eligible=True)

    result = manager.update([0.01, 0.01])

    self.assertLessEqual(
        result.projection_advance_m,
        np.linalg.norm([0.01, 0.01]) + result.diffusion_spacing_m + 1e-9,
    )
    np.testing.assert_array_equal(result.active_traj[0], [0.01, 0.01])


def test_fallback_prunes_passed_diffusion_points_without_reversing(self):
    manager = self.make_manager(point_spacing=0.05)
    candidate = np.array(
        [[0.30, 0.0], [0.45, 0.0], [0.60, 0.0], [0.75, 0.0]]
    )
    manager.update([0.0, 0.0], candidate, candidate_eligible=True)

    result = manager.update([0.47, 0.0])

    np.testing.assert_array_equal(
        result.active_traj[-2:],
        [[0.60, 0.0], [0.75, 0.0]],
    )
    self.assertEqual(result.diffusion_point_count, 2)
    self.assertTrue(np.all(np.diff(result.active_traj[:, 0]) > 0.0))


def test_first_distinct_reference_is_forward_of_chassis(self):
    manager = self.make_manager(point_spacing=0.05)
    candidate = np.array(
        [[0.60, 0.0], [0.75, 0.0], [0.90, 0.0]]
    )
    manager.update([0.0, 0.0], candidate, candidate_eligible=True)

    result = manager.update([0.02, 0.03])
    first = next(
        point
        for point in result.active_traj[1:]
        if np.linalg.norm(point - result.active_traj[0]) > 1e-9
    )

    self.assertGreater(np.dot(first - result.active_traj[0], [1.0, 0.0]), 0.0)
```

- [ ] **Step 2: Run the focused tests and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  TrajectoryManagerTests.test_chassis_projection_cannot_jump_across_self_crossing_centerline \
  TrajectoryManagerTests.test_fallback_prunes_passed_diffusion_points_without_reversing \
  TrajectoryManagerTests.test_first_distinct_reference_is_forward_of_chassis \
  -v
```

Expected: metadata fields do not exist and unrestricted closest projection can
select the later crossing branch.

- [ ] **Step 3: Extend immutable update metadata**

Add fields to `TrajectoryUpdate`:

```python
diffusion_spacing_m: float
projection_advance_m: float
blind_max_turn_degrees: float
```

Return zero values when no history is active. Compute
`blind_max_turn_degrees` over:

```python
seam = np.vstack(
    (
        active[: 1 + blind_point_count],
        diffusion_tail[0],
    )
)
```

Ignore duplicate consecutive points before calculating segment headings.

- [ ] **Step 4: Implement bounded centerline projection**

Add:

```python
def _advance_centerline(self, chassis: np.ndarray):
    if self._history_centerline is None:
        return False, 0.0
    displacement = (
        0.0
        if self._last_chassis is None
        else float(np.linalg.norm(chassis - self._last_chassis))
    )
    maximum_arc = min(
        self._polyline_length(self._history_centerline),
        displacement + self._diffusion_spacing,
    )
    distance, _, _, projection_arc = self._projection_with_arc(
        self._history_centerline,
        self._cumulative_lengths(self._history_centerline),
        chassis,
        maximum_arc=maximum_arc,
    )
    del distance
    self._history_centerline = self._trim_from_arc(
        self._history_centerline,
        projection_arc,
    )
    self._advance_diffusion_suffix(projection_arc)
    self._last_chassis = chassis.copy()
    return True, projection_arc
```

Extend `_projection_with_arc(..., maximum_arc=None)` so it truncates the last
search segment at `maximum_arc` and never examines later segments.

Because the centerline is physically trimmed after every update, the next
projection starts at prior progress and cannot move backward.

- [ ] **Step 5: Prune passed fallback diffusion points**

Before trimming, calculate progress relative to `_diffusion_start_arc`.

If `projection_arc <= _diffusion_start_arc`, subtract it from the start arc and
retain the full suffix.

Otherwise:

```python
passed_in_diffusion = projection_arc - self._diffusion_start_arc
candidate_cumulative = self._cumulative_lengths(self._diffusion_tail)
keep_index = int(
    np.searchsorted(
        candidate_cumulative,
        passed_in_diffusion,
        side="right",
    )
)
if keep_index >= len(self._diffusion_tail):
    self._clear_persistent_state()
else:
    next_point_arc = candidate_cumulative[keep_index]
    self._diffusion_tail = self._diffusion_tail[keep_index:].copy()
    self._diffusion_start_arc = next_point_arc - passed_in_diffusion
```

The centerline retains the projected segment to the first remaining exact
diffusion point. That interval is sampled as blind geometry; passed exact
diffusion points never reappear behind the chassis.

- [ ] **Step 6: Enforce the first-distinct-reference forward invariant**

In transient active construction, use the initial centerline tangent:

```python
tangent = self._history_centerline[1] - self._history_centerline[0]
forward_blind = [
    point
    for point in blind_guides
    if np.dot(point - chassis, tangent) > 0.0
]
```

Check the first distinct point of
`forward_blind + diffusion_tail`. If no distinct forward point exists, return
no active trajectory and use `history_exhausted`. For a new candidate, reject
transactionally with `candidate_nonforward` and retain the advanced prior
centerline.

- [ ] **Step 7: Run manager tests and commit**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  TrajectoryManagerTests -v
```

Then:

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git diff --cached --name-only
git commit -m "fix: enforce monotonic blind reference progress"
```

Expected staged names: only the two Task 2 files; `controllers.py` must not be
listed.

### Task 3: Expose spacing, progress, and turn diagnostics

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py:840-900`
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py:1780-1840`

**Interfaces:**
- Consumes: Task 2 `TrajectoryUpdate` diagnostic fields.
- Produces: JSONL keys `trajectory_diffusion_spacing_m`, `trajectory_projection_advance_m`, and `trajectory_blind_max_turn_deg`.

- [ ] **Step 1: Write failing client source diagnostics test**

Extend `test_client_records_candidate_decision_and_active_trajectory` with:

```python
for required in (
    '"trajectory_diffusion_spacing_m"',
    "trajectory_update.diffusion_spacing_m",
    '"trajectory_projection_advance_m"',
    "trajectory_update.projection_advance_m",
    '"trajectory_blind_max_turn_deg"',
    "trajectory_update.blind_max_turn_degrees",
):
    self.assertIn(required, source)
```

- [ ] **Step 2: Run the focused source test and verify red**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
  RosClientSourceTests.test_client_records_candidate_decision_and_active_trajectory \
  -v
```

Expected: all three new JSONL names are absent.

- [ ] **Step 3: Add plan JSONL and information-log fields**

In the `"type": "plan"` record add:

```python
"trajectory_diffusion_spacing_m": (
    trajectory_update.diffusion_spacing_m
),
"trajectory_projection_advance_m": (
    trajectory_update.projection_advance_m
),
"trajectory_blind_max_turn_deg": (
    trajectory_update.blind_max_turn_degrees
),
```

Add `spacing`, `advance`, and `blind_turn` to the active-trajectory information
log using three-decimal formatting. Do not change MPC construction or update
arguments.

- [ ] **Step 4: Run client/core and BEV regressions**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_wheeled_client_core.py' -v
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_rgb_bev_visualizer.py' -v
```

Expected: both suites pass, including virtual-camera reprojection and selected
diffusion visualization tests.

- [ ] **Step 5: Commit diagnostics**

```bash
git add \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git diff --cached --name-only
git commit -m "feat: diagnose monotonic blind references"
```

Do not stage `scripts/realworld/controllers.py`.

### Task 4: Replay the angular-jump record and complete verification

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: Tasks 1-3 and the read-only latest JSONL log.
- Produces: evidence that guide count, centerline turns, latency, and tracked regressions satisfy the design.

- [ ] **Step 1: Repeat the 100-cycle synthetic stability test five times**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
for run in 1 2 3 4 5; do
  PYTHONPATH=".:${PYTHONPATH}" python tests/test_wheeled_client_core.py \
    TrajectoryManagerTests.test_lateral_reanchoring_never_accumulates_projection_corners \
    -q || exit 1
done
```

Expected: all five runs pass with bounded blind count and no accumulated
right-angle turns.

- [ ] **Step 2: Replay the latest record through a fresh manager**

Run this from the repository root:

```bash
PYTHONPATH="navdp_runtime/navdp-imagegoal-client:${PYTHONPATH}" python - <<'PY'
import json
from pathlib import Path

import numpy as np

from utils_tasks.wheeled_client_core import TrajectoryManager

path = Path(
    "NavDP-official-bebb436/navdp_logs/20260728_031428_mpc.jsonl"
)
manager = TrajectoryManager()
accepted = []
maximum_count_ratio = 0.0
maximum_turn = 0.0
maximum_ms = 0.0
for line in path.open(encoding="utf-8"):
    record = json.loads(line)
    if record.get("type") != "plan":
        continue
    chassis = np.asarray(record["snapshot_odom"][:2], dtype=np.float64)
    raw_candidate = record.get("candidate_world_xy")
    candidate = (
        None
        if raw_candidate is None
        else np.asarray(raw_candidate, dtype=np.float64)
    )
    eligible = (
        candidate is not None
        and record.get("critic") is not None
        and record["critic"] >= -3.0
    )
    update = manager.update(
        chassis,
        candidate,
        candidate_eligible=eligible,
    )
    if update.active_traj is not None:
        np.testing.assert_array_equal(update.active_traj[0], chassis)
    if update.candidate_accepted:
        normalized = manager._normalize_polyline(candidate)
        np.testing.assert_array_equal(
            update.active_traj[-len(normalized) :],
            normalized,
        )
        physical_slots = max(
            1.0,
            update.blind_length_m / update.diffusion_spacing_m,
        )
        ratio = update.blind_point_count / physical_slots
        maximum_count_ratio = max(maximum_count_ratio, ratio)
        accepted.append(
            (
                record["plan_id"],
                update.blind_point_count,
                update.diffusion_point_count,
                update.mpc_prediction_steps,
            )
        )
    maximum_turn = max(
        maximum_turn,
        update.blind_max_turn_degrees,
    )
    maximum_ms = max(maximum_ms, update.manager_update_ms)

assert accepted
assert maximum_count_ratio <= 1.25
assert maximum_turn < 60.0
assert maximum_ms < 200.0
print("accepted", accepted)
print("maximum_count_ratio", round(maximum_count_ratio, 3))
print("maximum_turn_deg", round(maximum_turn, 3))
print("maximum_manager_ms", round(maximum_ms, 3))
PY
```

Expected: accepted candidate tails match exactly, count ratio stays at or below
`1.25`, no reproduced 90-degree blind ladder, and manager updates remain under
`200 ms`.

- [ ] **Step 3: Run fresh complete relevant verification**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_controllers.py'
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_wheeled_client_core.py'
PYTHONPATH=".:${PYTHONPATH}" python -m unittest discover \
  -s tests -p 'test_rgb_bev_visualizer.py'
python -m py_compile \
  utils_tasks/wheeled_client_core.py \
  scripts/realworld/controllers.py \
  scripts/realworld/navdp_imagegoal_client.py
cd /home/dev/navdp_deployment
git diff --check
git status --short --untracked-files=no
```

Expected: all relevant tests and compilation pass. The only pre-existing
tracked modification outside committed task files is the user's unstaged
`scripts/realworld/controllers.py` weight adjustment.

- [ ] **Step 4: Request read-only final review**

Review the implementation against:

```text
docs/superpowers/specs/2026-07-28-monotonic-blind-centerline-design.md
```

The reviewer must inspect the implementation range read-only and specifically
check:

- no chassis-projection connector enters persistent state;
- bounded projection cannot move backward or jump across the crossing test;
- fallback pruning cannot return passed diffusion points;
- candidate tails remain exact;
- the dirty controller weight adjustment was preserved and excluded.

Fix every Critical or Important finding with a failing regression test before
declaring completion.
