# Restore Upstream NavDP MPC Design

## Goal

Restore the trajectory tracking semantics from
`InternRobotics/NavDP@bebb436a9856acbd6ed2a63234a99db6bac2fd3a`
without removing the deployment improvements in this repository.

The restored data flow is:

```text
D435 RGB-D
  -> NavDP diffusion inference
  -> virtual-camera ground reprojection
  -> selected reprojected diffusion trajectory
  -> upstream NavDP MPC
  -> chassis command
```

There is no persistent trajectory history, blind-zone guide generation, or
`TrajectoryManager` between the reprojected diffusion result and MPC.

## Preserved Deployment Improvements

The rollback must preserve:

- D435 input and calibrated camera transforms.
- Virtual-camera ground reprojection.
- The current ROS 2 client, control gate, deadman behavior, and shutdown flow.
- Critic threshold safety handling.
- Goal verification and arrival handling.
- Raw and reprojected trajectory diagnostics.
- RGB, BEV, and MPC video recording.
- Selected-diffusion provenance and visualization.
- Existing robot-specific velocity arguments and ROS topics.

These components may be adapted only where their trajectory input changes
from `active_traj` to the directly selected reprojected diffusion trajectory.

## Upstream MPC Semantics

Port the MPC behavior from upstream
`utils_tasks/tracking_utils.py::MPC_Controller`:

- Fixed prediction horizon `N=15`.
- Default `desired_v=0.5`, `v_max=0.5`, and `w_max=0.5`; the deployment client
  continues passing its configured maximum velocity values.
- `ref_gap=3`.
- Time step `T=0.1`.
- Pose state `(x, y, theta)` and controls `(v, w)`.
- Cost weights:
  - `Q = diag([10.0, 10.0, 0.0])`
  - `R = diag([0.02, 0.15])`
- Reference trajectory length `N // ref_gap + 1`.
- The same 50x index-linear densification.
- The same nearest-point and metric-distance reference selection.
- Reference yaw is zero, as in upstream.
- Rebuild the MPC from each newly accepted reprojected diffusion trajectory,
  matching upstream asynchronous planning behavior.

`N` is the MPC control horizon. It is not the number of points output by the
diffusion model and does not truncate or resample that trajectory.

## Removed Trajectory-Manager Behavior

Remove the runtime use of:

- Persistent historical centerlines.
- Blind guide points.
- Candidate join-distance and join-heading replacement gates.
- Bounded centerline projection and fallback suffix pruning.
- Dynamic `blind_steps`.
- Dynamic `N=diffusion_point_count`.
- `prediction_steps = blind_steps + diffusion_point_count`.
- Trajectory-manager CLI options and runtime diagnostics.

The utility implementation and tests should also be removed when no remaining
runtime or test code imports them.

## Client Installation and Failure Behavior

When planning succeeds, reprojection is valid, and the critic is safe:

1. Select the reprojected diffusion trajectory.
2. Validate that it is finite and contains at least two distinct XY points.
3. Construct a fresh upstream-semantics MPC with that entire trajectory.
4. Install an immutable copy as the currently tracked and visualized path.
5. Commit selected-diffusion provenance only after successful MPC creation.

When any of those steps fail:

- Do not install a partial trajectory.
- Mark trajectory control unavailable.
- Preserve the existing fail-closed control behavior.

There is no historical fallback trajectory after a rejected or missing plan.

## Visualization and Diagnostics

The MPC BEV guide layer must show the directly installed reprojected diffusion
trajectory. It must not prepend the chassis point or generate extra blind
points.

Keep raw/reprojected geometry and MPC/control diagnostics. Remove fields whose
only meaning came from `TrajectoryManager`, including blind length/count,
diffusion spacing, projection advance, join distance, manager update time, and
dynamic MPC prediction-step counts.

The installed trajectory point count may remain the model output count, such
as 24 points. The MPC prediction horizon remains 15 independently.

## Verification

Tests must prove:

1. The controller uses upstream `N=15`, `ref_gap=3`, `Q`, and `R`.
2. The controller has no `blind_steps` or dynamic horizon.
3. A full reprojected diffusion trajectory is passed directly to MPC.
4. No `TrajectoryManager` is constructed or called by the client.
5. Reprojection, D435, BEV, video, deadman, and arrival tests remain green.
6. The installed visualization contains no generated blind points.
7. A recent JSONL plan replays as a direct reprojected trajectory with a fixed
   15-step MPC horizon.

## Out of Scope

- Changes to NavDP inference or its output point count.
- Diffusion resampling to 10 or 15 points.
- Reintroducing blind-zone filling.
- Changing camera calibration or reprojection math.
- Changing robot topics, startup posture, arrival verification, or video
  encoding.
