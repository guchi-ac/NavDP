# NavDP Trajectory Manager Design

## Goal

Insert a `TrajectoryManager` between the InternNav/NavDP model output and the
MPC controller. The manager maintains stable historical guide points in the
odometry frame so every MPC reference starts at the current chassis position,
even though the front D435 does not provide a useful model guide point near
the chassis.

The model-selected trajectory is a candidate update. It must not directly
replace the trajectory followed by MPC.

## Scope

This change applies to the real-world image-goal client in
`navdp_runtime/navdp-imagegoal-client`. It does not change the NavDP server,
model inference, candidate critic calculation, MPC objective, or robot
interface.

The existing model candidate visualization remains available. The red tracked
trajectory changes to the manager's actual `active_traj`.

## Coordinate Frame and State

`TrajectoryManager` is a ROS-independent class in
`utils_tasks/wheeled_client_core.py`. It stores `active_traj` as an `N x 2`
NumPy array in the odometry frame. It never stores camera-frame points.

Each update consumes:

- the current chassis position in the odometry frame;
- an optional candidate trajectory in the odometry frame; and
- whether the candidate passed the existing critic threshold.

Each update returns an immutable result containing:

- the executable `active_traj`, or no trajectory;
- whether the candidate was accepted;
- a machine-readable reason;
- the candidate-to-history join distance, when a join was attempted; and
- the remaining active trajectory length.

The returned `active_traj` always has at least two distinct points and its
first point is exactly the current chassis position. The manager owns its
internal copy so callers cannot mutate its state.

## Active-Trajectory Update

### Advancing History

At the beginning of every planning cycle, project the current chassis position
onto the segments of the existing `active_traj`. Remove the portion already
traversed and retain the forward suffix. Determine exhaustion from the
historical arc between that projection and the terminal point; do not count
the chassis-to-projection correction distance as remaining history. Build an
executable suffix from:

1. the exact current chassis position;
2. the closest projected point when it is distinct from the chassis; and
3. the untraversed guide points after that segment.

Remove consecutive duplicate points and resample the result at a default
spacing of `0.05 m`. If the remaining arc length is less than `0.20 m`, the
history is exhausted and is no longer executable.

### First Accepted Candidate

When no executable history exists and a critic-qualified candidate is valid,
prepend the current chassis position to the candidate and linearly resample
the complete polyline at `0.05 m`. This explicitly fills the near-field gap
between the chassis and the model's first guide point.

### Joining a New Candidate

When executable history exists, treat a critic-qualified model trajectory as
a far-field candidate:

1. Find the closest point on the forward historical polyline to the
   candidate's first point.
2. Measure the Euclidean join distance and both heading changes through the
   connector: history-to-connector and connector-to-candidate. Treat a
   connector no longer than one resampling interval as coincident by snapping
   the candidate's first point onto the historical join point.
3. Accept the candidate only when the join distance is no more than `0.50 m`
   and the absolute heading change is no more than `60 degrees`.
4. Retain the historical path from the chassis through the join point.
5. Append the candidate from its first point onward.
6. Remove duplicate points and resample the combined polyline at `0.05 m`.

This keeps the near-field guide points stable while allowing the model to
refresh the route beyond the D435's useful near field. The four numeric
limits are exposed as command-line arguments so robot logs can be used to tune
them without code changes.

If the candidate fails validation or either continuity check, keep the
advanced historical suffix unchanged. A rejected candidate never partially
mutates `active_traj`.

## Candidate and Failure Semantics

A model candidate below the existing critic threshold is not offered for
joining. A single low-critic inference therefore does not stop the robot when
an executable historical suffix remains.

Malformed candidate geometry, non-finite values, a transient model request
failure, or a discontinuous candidate also leave valid history intact. The
manager continues advancing that history from fresh odometry until it is
exhausted.

Control is allowed only when all existing frame, odometry, arrival, explicit
enable-control, and timeout gates pass and the manager has an executable
`active_traj`. If no active trajectory has ever been established, or the
history is exhausted without an acceptable candidate, the command is zero
with reason `trajectory_missing`.

The active-plan timestamp is refreshed whenever a planning cycle successfully
advances or joins an executable `active_traj`. It is not refreshed when there
is no executable trajectory. Existing frame and odometry deadman timers remain
unchanged.

## ROS Client Integration

The planning loop keeps the existing camera-to-odometry transformation for
the selected model trajectory and all visualization candidates. It then:

1. advances the manager using the frame-synchronized chassis position;
2. submits the selected transformed model trajectory only when its critic is
   at or above the configured threshold;
3. sends the returned `active_traj` to `Mpc_controller`;
4. records manager state and the candidate decision in plan diagnostics; and
5. publishes the `active_traj` as the authoritative red trajectory.

The MPC optimizer is not modified. Its initial state remains the latest
odometry pose, and its reference sampler receives a dense path whose first
point is the same chassis position.

The client exposes these defaults:

- `--trajectory-point-spacing 0.05`
- `--trajectory-join-distance 0.50`
- `--trajectory-join-heading-deg 60.0`
- `--trajectory-min-remaining 0.20`

## Diagnostics

Every plan diagnostic records:

- the raw selected model trajectory in world coordinates;
- the active trajectory passed to MPC;
- whether the candidate was accepted;
- the manager reason;
- join distance;
- remaining active arc length; and
- the model critic.

Informational logs distinguish initialization, accepted joins, retained
history, candidate rejection, and exhausted history. Candidate rejection is
rate-limited through the existing planning period and is not logged as a
control failure while valid history remains.

## Testing

Pure NumPy unit tests cover:

- initial interpolation from the exact chassis position to a distant first
  candidate point;
- output spacing and duplicate removal;
- projection-based removal of traversed history;
- preservation of near-field historical points during an accepted join;
- rejection by join distance;
- rejection by heading discontinuity;
- retention of history for a low-critic or malformed candidate;
- exhaustion below the minimum remaining length;
- immutable/copy-safe manager results; and
- finite `N x 2`, at-least-two-point output invariants.

Client integration tests verify:

- the transformed model trajectory is submitted to `TrajectoryManager`
  instead of directly to `Mpc_controller`;
- the MPC update and authoritative visualization use `active_traj`;
- a low-critic candidate does not invalidate executable history;
- `trajectory_missing` blocks control when no active trajectory exists;
- the new command-line defaults; and
- diagnostic fields for the candidate decision and active trajectory.

The focused unit suite, the complete existing client test suite, Python
compilation, and `git diff --check` must pass. No control-enabled robot run is
part of local verification.
