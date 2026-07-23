# NavDP Trajectory Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maintain odometry-frame historical guide points and provide MPC with a stable active trajectory whose first point is the current chassis position.

**Architecture:** Add a ROS-independent `TrajectoryManager` to the existing client core. The ROS planning loop advances retained history on every fresh frame, treats critic-qualified InternNav/NavDP output as a far-field candidate, and sends only the manager's `active_traj` to MPC and visualization.

**Tech Stack:** Python 3.10, NumPy, ROS2 Humble, `unittest`, CasADi MPC.

## Global Constraints

- Keep the NavDP server, model inference, critic calculation, MPC objective, and robot interfaces unchanged.
- Store manager state only in the odometry frame.
- Make `active_traj[0]` exactly equal to the current chassis position.
- Resample guide points at `0.05 m` by default.
- Accept joins at no more than `0.50 m` and `60 degrees` by default.
- Treat less than `0.20 m` of remaining path as exhausted.
- Continue valid history through low-critic, malformed, discontinuous, or transiently unavailable candidates.
- Do not add dependencies or enable robot control during verification.

---

### Task 1: TrajectoryManager Geometry and State

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`

**Interfaces:**
- Produces: `TrajectoryUpdate(active_traj, candidate_accepted, reason, join_distance_m, remaining_length_m)`.
- Produces: `TrajectoryManager(point_spacing=0.05, join_distance=0.50, join_heading_degrees=60.0, min_remaining=0.20)`.
- Produces: `TrajectoryManager.update(chassis_xy, candidate_world_xy=None, candidate_eligible=False) -> TrajectoryUpdate`.

- [ ] **Step 1: Add failing initialization and spacing tests**

Add a `TrajectoryManagerTests` class that imports the manager through
`client_core` and verifies the exact chassis anchor and 5 cm spacing:

```python
class TrajectoryManagerTests(unittest.TestCase):
    def make_manager(self, **changes):
        values = dict(
            point_spacing=0.05,
            join_distance=0.50,
            join_heading_degrees=60.0,
            min_remaining=0.20,
        )
        values.update(changes)
        return client_core.TrajectoryManager(**values)

    def test_initial_candidate_starts_at_chassis_and_fills_near_field(self):
        manager = self.make_manager()
        result = manager.update(
            np.array([2.0, 3.0]),
            np.array([[3.0, 3.0], [4.0, 3.0]]),
            candidate_eligible=True,
        )

        self.assertTrue(result.candidate_accepted)
        self.assertEqual(result.reason, "initialized")
        np.testing.assert_array_equal(result.active_traj[0], [2.0, 3.0])
        distances = np.linalg.norm(np.diff(result.active_traj, axis=0), axis=1)
        np.testing.assert_allclose(distances, 0.05, atol=1e-9)
```

- [ ] **Step 2: Run the initialization test and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_wheeled_client_core.TrajectoryManagerTests.test_initial_candidate_starts_at_chassis_and_fills_near_field -v
```

Expected: FAIL because `TrajectoryManager` does not exist.

- [ ] **Step 3: Add failing history, join, rejection, and exhaustion tests**

Cover these independent behaviors with separate tests:

```python
def test_advance_prunes_traversed_history_and_reanchors_at_chassis(self):
    manager = self.make_manager()
    manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update([0.35, 0.02])
    np.testing.assert_array_equal(result.active_traj[0], [0.35, 0.02])
    self.assertGreater(result.active_traj[-1, 0], 1.9)

def test_join_preserves_history_before_candidate(self):
    manager = self.make_manager()
    manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update(
        [0.10, 0.0],
        np.array([[1.05, 0.02], [2.0, 0.5]]),
        candidate_eligible=True,
    )
    self.assertTrue(result.candidate_accepted)
    np.testing.assert_array_equal(result.active_traj[0], [0.10, 0.0])
    self.assertTrue(np.any(np.isclose(result.active_traj[:, 0], 0.50, atol=0.03)))

def test_rejects_distant_candidate_without_mutating_history(self):
    manager = self.make_manager(join_distance=0.10)
    manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update(
        [0.10, 0.0],
        np.array([[1.0, 1.0], [2.0, 1.0]]),
        candidate_eligible=True,
    )
    self.assertFalse(result.candidate_accepted)
    self.assertEqual(result.reason, "join_distance")
    self.assertLess(np.max(np.abs(result.active_traj[:, 1])), 0.11)

def test_rejects_heading_discontinuity(self):
    manager = self.make_manager(join_heading_degrees=30.0)
    manager.update(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update(
        [0.10, 0.0],
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        candidate_eligible=True,
    )
    self.assertFalse(result.candidate_accepted)
    self.assertEqual(result.reason, "join_heading")

def test_low_critic_candidate_retains_history_until_exhausted(self):
    manager = self.make_manager(min_remaining=0.20)
    manager.update(
        [0.0, 0.0],
        np.array([[0.5, 0.0], [1.0, 0.0]]),
        candidate_eligible=True,
    )
    retained = manager.update(
        [0.50, 0.0],
        np.array([[0.5, 1.0], [1.0, 1.0]]),
        candidate_eligible=False,
    )
    self.assertEqual(retained.reason, "candidate_low_critic")
    self.assertIsNotNone(retained.active_traj)
    exhausted = manager.update([0.95, 0.0])
    self.assertIsNone(exhausted.active_traj)
    self.assertEqual(exhausted.reason, "history_exhausted")

def test_invalid_candidate_retains_history(self):
    manager = self.make_manager()
    manager.update(
        [0.0, 0.0],
        np.array([[0.5, 0.0], [1.0, 0.0]]),
        candidate_eligible=True,
    )
    result = manager.update(
        [0.10, 0.0],
        np.array([[math.nan, 0.0], [1.0, 0.0]]),
        candidate_eligible=True,
    )
    self.assertEqual(result.reason, "candidate_invalid")
    self.assertIsNotNone(result.active_traj)

def test_result_does_not_allow_mutating_manager_state(self):
    manager = self.make_manager()
    first = manager.update(
        [0.0, 0.0],
        np.array([[0.5, 0.0], [1.0, 0.0]]),
        candidate_eligible=True,
    )
    with self.assertRaises(ValueError):
        first.active_traj[0] = [99.0, 99.0]
    second = manager.update([0.10, 0.0])
    self.assertLess(second.active_traj[0, 0], 1.0)
```

- [ ] **Step 4: Run all manager tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_wheeled_client_core.TrajectoryManagerTests -v
```

Expected: FAIL only because the manager interfaces are missing.

- [ ] **Step 5: Implement the manager**

Add a frozen result dataclass and focused private helpers:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryUpdate:
    active_traj: Optional[np.ndarray]
    candidate_accepted: bool
    reason: str
    join_distance_m: Optional[float]
    remaining_length_m: float


class TrajectoryManager:
    def __init__(
        self,
        *,
        point_spacing=0.05,
        join_distance=0.50,
        join_heading_degrees=60.0,
        min_remaining=0.20,
    ):
        values = {
            "point_spacing": point_spacing,
            "join_distance": join_distance,
            "join_heading_degrees": join_heading_degrees,
            "min_remaining": min_remaining,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        self.point_spacing = float(point_spacing)
        self.join_distance = float(join_distance)
        self.join_heading = math.radians(float(join_heading_degrees))
        self.min_remaining = float(min_remaining)
        self._active_traj = None

    def update(
        self,
        chassis_xy,
        candidate_world_xy=None,
        candidate_eligible=False,
    ):
        chassis = np.asarray(chassis_xy, dtype=np.float64)
        if chassis.shape != (2,) or not np.isfinite(chassis).all():
            raise ValueError("chassis_xy must be a finite shape-(2,) point")

        history, had_history = self._advance_history(chassis)
        active = history
        accepted = False
        join_distance_m = None
        reason = "history_retained" if history is not None else "no_history"

        candidate = None
        if candidate_world_xy is not None:
            if not candidate_eligible:
                reason = "candidate_low_critic"
            else:
                try:
                    candidate = self._normalize_polyline(candidate_world_xy)
                except ValueError:
                    reason = "candidate_invalid"

        if candidate is not None:
            if history is None:
                initialized = self._resample_polyline(
                    self._normalize_polyline(np.vstack((chassis, candidate)))
                )
                if self._polyline_length(initialized) >= self.min_remaining:
                    active = initialized
                    accepted = True
                    reason = "initialized"
                else:
                    reason = "candidate_too_short"
            else:
                (
                    join_distance_m,
                    segment_index,
                    join_point,
                ) = self._closest_projection(history, candidate[0])
                if join_distance_m > self.join_distance:
                    reason = "join_distance"
                else:
                    history_heading = (
                        history[segment_index + 1] - history[segment_index]
                    )
                    candidate_heading = candidate[1] - candidate[0]
                    if (
                        self._heading_delta(history_heading, candidate_heading)
                        > self.join_heading
                    ):
                        reason = "join_heading"
                    else:
                        combined = self._normalize_polyline(
                            np.vstack(
                                (
                                    history[: segment_index + 1],
                                    join_point,
                                    candidate,
                                )
                            )
                        )
                        active = self._resample_polyline(combined)
                        accepted = True
                        reason = "candidate_joined"

        if active is None and had_history and candidate_world_xy is None:
            reason = "history_exhausted"
        self._active_traj = None if active is None else active.copy()
        remaining = (
            0.0
            if self._active_traj is None
            else self._polyline_length(self._active_traj)
        )
        result_traj = (
            None if self._active_traj is None else self._active_traj.copy()
        )
        if result_traj is not None:
            result_traj.setflags(write=False)
        return TrajectoryUpdate(
            active_traj=result_traj,
            candidate_accepted=accepted,
            reason=reason,
            join_distance_m=join_distance_m,
            remaining_length_m=remaining,
        )

    @staticmethod
    def _normalize_polyline(points):
        points = np.asarray(points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            raise ValueError("trajectory must be finite with shape (N, 2), N >= 2")
        keep = np.concatenate(
            (
                np.array([True]),
                np.linalg.norm(np.diff(points, axis=0), axis=1)
                > np.finfo(np.float64).eps,
            )
        )
        normalized = points[keep]
        if len(normalized) < 2:
            raise ValueError("trajectory must contain two distinct points")
        return normalized

    @staticmethod
    def _polyline_length(points):
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def _resample_polyline(self, points):
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total = float(cumulative[-1])
        targets = np.arange(0.0, total, self.point_spacing)
        if len(targets) == 0 or not np.isclose(targets[-1], total):
            targets = np.append(targets, total)
        else:
            targets[-1] = total
        return np.column_stack(
            (
                np.interp(targets, cumulative, points[:, 0]),
                np.interp(targets, cumulative, points[:, 1]),
            )
        )

    @staticmethod
    def _closest_projection(polyline, point):
        best = (math.inf, 0, polyline[0])
        for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
            segment = end - start
            fraction = float(
                np.clip(
                    np.dot(point - start, segment) / np.dot(segment, segment),
                    0.0,
                    1.0,
                )
            )
            projection = start + fraction * segment
            distance = float(np.linalg.norm(point - projection))
            if distance < best[0]:
                best = (distance, index, projection)
        return best

    def _advance_history(self, chassis):
        if self._active_traj is None:
            return None, False
        _, segment_index, projection = self._closest_projection(
            self._active_traj,
            chassis,
        )
        try:
            remainder = self._normalize_polyline(
                np.vstack(
                    (
                        chassis,
                        projection,
                        self._active_traj[segment_index + 1 :],
                    )
                )
            )
        except ValueError:
            return None, True
        if self._polyline_length(remainder) < self.min_remaining:
            return None, True
        advanced = self._resample_polyline(remainder)
        advanced[0] = chassis
        return advanced, True

    @staticmethod
    def _heading_delta(first, second):
        first_angle = math.atan2(first[1], first[0])
        second_angle = math.atan2(second[1], second[0])
        return abs(
            math.atan2(
                math.sin(second_angle - first_angle),
                math.cos(second_angle - first_angle),
            )
        )
```

When implementing, keep the helpers exactly focused on validation,
projection, resampling, and heading comparison. Resampling must include both
endpoints, produce no consecutive duplicates, and set the first row back to
the exact input chassis coordinates after interpolation. Return read-only
copies in `TrajectoryUpdate`.

- [ ] **Step 6: Run manager tests and verify GREEN**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_wheeled_client_core.TrajectoryManagerTests -v
```

Expected: all manager tests PASS.

- [ ] **Step 7: Commit the manager**

```bash
git add navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: add trajectory manager"
```

### Task 2: Active-Trajectory Control Gate

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py`

**Interfaces:**
- Changes: `control_stop_reason(*, now, enable_control, arrival_blocked, trajectory_ready, last_frame_time, last_odom_time, last_plan_time, frame_timeout, odom_timeout, plan_timeout) -> Optional[str]`.
- Removes: latest-candidate `critic_safe` as a direct control gate.

- [ ] **Step 1: Change deadman tests to the active-trajectory contract**

Change the test fixture from `critic_safe=True` to `trajectory_ready=True`.
Replace the low-critic test with:

```python
def test_stops_when_active_trajectory_is_missing(self):
    self.assertEqual(
        self.reason(trajectory_ready=False),
        "trajectory_missing",
    )
```

Keep arrival precedence by testing `arrival_blocked=True` together with
`trajectory_ready=False`.

- [ ] **Step 2: Run the deadman tests and verify RED**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest test_wheeled_client_core.ControlDeadmanTests -v
```

Expected: FAIL because the function still accepts `critic_safe`.

- [ ] **Step 3: Implement the new control gate**

Change the signature and gate:

```python
def control_stop_reason(
    *,
    now: float,
    enable_control: bool,
    arrival_blocked: bool,
    trajectory_ready: bool,
    last_frame_time: Optional[float],
    last_odom_time: Optional[float],
    last_plan_time: Optional[float],
    frame_timeout: float,
    odom_timeout: float,
    plan_timeout: float,
) -> Optional[str]:
    if not enable_control:
        return "control_disabled"
    if arrival_blocked:
        return "arrival"
    if not trajectory_ready:
        return "trajectory_missing"
    for name, timestamp, timeout in (
        ("frame", last_frame_time, frame_timeout),
        ("odom", last_odom_time, odom_timeout),
        ("plan", last_plan_time, plan_timeout),
    ):
        if timestamp is None:
            return f"{name}_missing"
        if now - timestamp > timeout:
            return f"{name}_stale"
    return None
```

- [ ] **Step 4: Run the deadman and manager tests**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.ControlDeadmanTests \
  test_wheeled_client_core.TrajectoryManagerTests -v
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the gate**

```bash
git add navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "refactor: gate control on active trajectory"
```

### Task 3: ROS Planning-Loop Integration

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py`
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py`

**Interfaces:**
- Consumes: `TrajectoryManager.update(chassis_xy, candidate_world_xy=None, candidate_eligible=False) -> TrajectoryUpdate`.
- Produces: `self.trajectory_ready`.
- Produces CLI: `--trajectory-point-spacing`, `--trajectory-join-distance`, `--trajectory-join-heading-deg`, `--trajectory-min-remaining`.

- [ ] **Step 1: Add failing source-integration tests**

Add focused source assertions:

```python
@staticmethod
def client_source():
    client_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "realworld"
        / "navdp_imagegoal_client.py"
    )
    return client_path.read_text(encoding="utf-8")

def test_client_routes_model_candidate_through_trajectory_manager(self):
    source = self.client_source()
    self.assertIn("TrajectoryManager(", source)
    self.assertIn("candidate_eligible=critic_safe", source)
    self.assertIn("active_traj = trajectory_update.active_traj", source)
    self.assertIn("self.mpc.update_ref_traj(active_traj)", source)
    self.assertNotIn("self.mpc.update_ref_traj(world_xy)", source)

def test_client_gates_control_on_active_trajectory(self):
    source = self.client_source()
    self.assertIn("self.trajectory_ready = False", source)
    self.assertIn("trajectory_ready=self.trajectory_ready", source)
    self.assertNotIn("critic_safe=self.critic_safe", source)

def test_client_exposes_trajectory_manager_defaults(self):
    source = self.client_source()
    for required in (
        '--trajectory-point-spacing", type=float, default=0.05',
        '--trajectory-join-distance", type=float, default=0.50',
        '--trajectory-join-heading-deg", type=float, default=60.0',
        '--trajectory-min-remaining", type=float, default=0.20',
    ):
        self.assertIn(required, source)
```

Use the existing test class's client path convention rather than importing
ROS packages.

- [ ] **Step 2: Run the new integration tests and verify RED**

Run each new test by full unittest name. Expected: FAIL because the client
still sends `world_xy` directly to MPC and gates on `critic_safe`.

- [ ] **Step 3: Instantiate and use TrajectoryManager**

Import `TrajectoryManager`, construct it from the four CLI values, and replace
`self.critic_safe` with `self.trajectory_ready`.

At each fresh planning snapshot:

```python
trajectory_update = self.trajectory_manager.update(
    snapshot.odom_xy_yaw[:2]
)
```

After transforming a model candidate:

```python
trajectory_update = self.trajectory_manager.update(
    snapshot.odom_xy_yaw[:2],
    retained_world_xy,
    candidate_eligible=critic_safe,
)
active_traj = trajectory_update.active_traj
```

If model inference or candidate processing fails, keep the already-advanced
`trajectory_update`. If it contains an active path, update/create MPC from
that path and refresh the active-plan timestamp. If it contains no active
path, set `self.trajectory_ready = False` and leave commands blocked.

- [ ] **Step 4: Route active trajectory to consumers and diagnostics**

Use `active_traj` for:

```python
self.mpc = Mpc_controller(
    active_traj,
    desired_v=self.args.max_v,
    v_max=self.args.max_v,
    w_max=self.args.max_w,
)
self.mpc.update_ref_traj(active_traj)
self.visualization_state = VisualizationState(
    trajectory=active_traj.copy(),
    trajectory_prefix=trajectory_prefix,
    candidates=candidate_world_xy.copy(),
    values=candidate_values.copy(),
    critic=critic_max,
    arrived=False,
)
```

Record these exact plan keys:

```python
"candidate_world_xy": retained_world_xy,
"active_traj": active_traj,
"candidate_accepted": trajectory_update.candidate_accepted,
"trajectory_reason": trajectory_update.reason,
"trajectory_join_distance_m": trajectory_update.join_distance_m,
"trajectory_remaining_length_m": trajectory_update.remaining_length_m,
```

Keep all transformed model candidates in the candidate overlay.

- [ ] **Step 5: Wire the active-trajectory deadman and CLI**

Pass `trajectory_ready=self.trajectory_ready` to `control_stop_reason` and
add the four parser arguments with the approved defaults.

- [ ] **Step 6: Run focused integration and core tests**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client/tests
PYTHONPATH=.. python3 -m unittest \
  test_wheeled_client_core.TrajectoryManagerTests \
  test_wheeled_client_core.ControlDeadmanTests \
  test_wheeled_client_core.RosClientSourceTests -v
```

Expected: all selected tests PASS.

- [ ] **Step 7: Commit the integration**

```bash
git add navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
git commit -m "feat: stabilize MPC guide trajectory"
```

### Task 4: Documentation and Complete Verification

**Files:**
- Modify: `navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md`

**Interfaces:**
- Documents: manager behavior, exact distance defaults, tuning flags, and diagnostics.

- [ ] **Step 1: Document runtime behavior**

Add a Chinese section stating:

- the closest active guide point is the chassis itself (`0 m`);
- subsequent manager points default to `0.05 m` spacing;
- the default MPC reference-state spacing is about `0.06 m` at `0.1 m/s`;
- low-critic or discontinuous candidates do not erase usable history;
- the robot stops after history is exhausted; and
- the four tuning arguments and their defaults.

- [ ] **Step 2: Run the complete local test suite**

Run:

```bash
cd navdp_runtime/navdp-imagegoal-client
python3 -m unittest discover -s tests -v
```

Expected: all tests PASS with zero failures and zero errors.

- [ ] **Step 3: Compile modified Python files**

Run:

```bash
python3 -m py_compile \
  navdp_runtime/navdp-imagegoal-client/utils_tasks/wheeled_client_core.py \
  navdp_runtime/navdp-imagegoal-client/scripts/realworld/navdp_imagegoal_client.py \
  navdp_runtime/navdp-imagegoal-client/tests/test_wheeled_client_core.py
```

Expected: exit status 0 and no output.

- [ ] **Step 4: Check the complete diff**

Run:

```bash
git diff --check HEAD
git status --short
git diff --stat HEAD
```

Expected: no whitespace errors; only the intended runtime, test, and
documentation paths are part of this feature.

- [ ] **Step 5: Commit documentation**

```bash
git add navdp_runtime/navdp-imagegoal-client/scripts/realworld/README_NAVDP_CLIENT_ZH.md
git commit -m "docs: explain trajectory manager"
```

- [ ] **Step 6: Re-run completion verification**

Re-run the full test suite, compilation command, `git diff --check HEAD^`, and
inspect `git log --oneline --decorate -5`. Record exact test counts and commit
IDs in the handoff.
