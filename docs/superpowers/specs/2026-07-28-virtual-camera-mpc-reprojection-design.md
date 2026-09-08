# Virtual-Camera MPC Reprojection Design

## Goal

Reproduce the official NavDP `z=-0.2 m` trajectory projection, map that
virtual image-space trajectory back onto the real robot ground plane, and use
the resulting path as the MPC control reference.

The change must preserve the original selected diffusion trajectory for
diagnostics. In the MPC RGB BEV:

- cyan remains the unmodified selected diffusion trajectory;
- yellow is the reprojected guide actually installed in MPC;
- red remains the MPC predicted state trajectory.

## Scope

This change applies to the ROS2 image-goal client. It does not modify the
NavDP server, checkpoint, critic, candidate sampling, RGB-D point-cloud
projection, or MPC objective weights.

The virtual camera height is configurable through
`--virtual-camera-height`, with a default of `0.2` metres.

## Geometry

### Stage 1: Official NavDP virtual projection

For each selected local waypoint `(forward_x, left_y)`, reproduce the
official NavDP visualization formula with virtual height `h_v`:

```text
u = fx * (-left_y / forward_x) + cx
v = image_height - 1 + fy * (h_v / forward_x) - cy
```

The implementation will use the unresized planning-frame intrinsic matrix and
image height. Points with non-positive forward distance or non-finite
coordinates are invalid.

### Stage 2: Real optical ray

Convert each virtual pixel `(u, v)` into an unnormalised ray in the real
camera optical frame:

```text
ray_camera = [(u - cx) / fx, (v - cy) / fy, 1]
```

Transform the ray direction and camera origin into `base_link` using the
planning frame's `base_from_camera` transform.

### Stage 3: Ground-plane intersection

Intersect each transformed ray with the `base_link` ground plane `z=0`:

```text
scale = -camera_origin_z / ray_base_z
point_base = camera_origin + scale * ray_base
```

A point is invalid when:

- the ray is parallel or nearly parallel to the ground;
- the intersection scale is non-positive;
- the result is non-finite;
- the resulting point is not in front of the robot;
- the reprojected polyline reverses its cumulative forward progress.

The whole candidate is rejected if any retained waypoint is invalid. Points
will not be silently dropped because changing the point count would also
change the MPC horizon and could hide a discontinuity.

### Stage 4: Odom conversion

The ground intersections are already expressed in `base_link`; convert them
to odom/world XY using only the current base odometry pose. Do not apply the
camera translation or yaw a second time.

## Planning Data Flow

For every successful NavDP response:

1. Keep the returned top-critic trajectory as `raw_local_xy`.
2. Convert it through the existing camera-aligned transform to
   `raw_world_xy` for the cyan diagnostic layer.
3. Reproject `raw_local_xy` through the virtual camera and real ground plane
   to produce `reprojected_base_xy`.
4. Convert `reprojected_base_xy` into `reprojected_world_xy`.
5. Pass the retained reprojected path to `TrajectoryManager`.
6. Install the resulting active guide into MPC.
7. If the candidate is accepted and MPC installation succeeds, snapshot the
   corresponding raw selected trajectory for the cyan layer.

Candidate acceptance, trajectory-history joining, critic threshold handling,
and MPC lifecycle otherwise remain unchanged.

## Failure Behaviour

Reprojection errors are candidate failures, not process failures.

- If a previous active guide remains valid, `TrajectoryManager` retains it.
- If no active guide exists, trajectory readiness is false and the control
  loop commands a stop.
- A rejected reprojection must never partially update MPC or the cyan
  provenance snapshot.
- Logs must include a concise rejection reason suitable for diagnosing ray,
  ground-intersection, and polyline-order failures.

This transformation is not an obstacle-clearance guarantee. It changes the
geometric interpretation of NavDP waypoints but does not turn unobserved
space into observed free space.

## Visualisation and Diagnostics

The existing BEV colours retain their meanings:

- cyan line: original selected diffusion in odom coordinates;
- yellow guide points: reprojected active trajectory consumed by MPC;
- red line: MPC prediction;
- green line: measured odometry history.

Each plan diagnostic record will contain:

- `selected_local_xy`;
- `raw_selected_world_xy`;
- `reprojected_base_xy`;
- `reprojected_world_xy`;
- `virtual_camera_height_m`;
- reprojection status or rejection reason.

These fields provide a direct comparison between the server output and the
control reference.

## Testing

Unit tests will cover:

1. exact agreement with the official `z=-0.2` pixel formula;
2. a synthetic level-camera case with analytically known ground
   intersections;
3. a pitched-camera case using the full optical transform;
4. rejection of non-positive forward coordinates;
5. rejection of parallel, upward, and behind-camera intersections;
6. rejection of non-finite and forward-reversing polylines;
7. base-to-odom conversion without a second camera transform;
8. planning data flow: raw selected provenance remains cyan while the
   reprojected path reaches `TrajectoryManager` and MPC;
9. retention of the previous guide when a new reprojection is invalid.

The focused geometry and client tests will be run first, followed by the
complete image-goal client test suite.

## Non-Goals

- Changing critic scoring or selecting a different diffusion sample.
- Adding occupancy, FOV, or footprint-clearance gating.
- Changing MPC weights, limits, or reference spacing.
- Modifying the NavDP server-side `fps_pointgoal` renderer.
- Reprojecting all 16 candidates; only the server-selected trajectory feeds
  control.
