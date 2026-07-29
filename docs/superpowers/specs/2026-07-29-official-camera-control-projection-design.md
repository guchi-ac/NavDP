# Official-Camera Control Projection Design

## Goal

Use the live D435 color-camera intrinsic matrix with NavDP's official
level-camera geometry for the selected-to-guide control transformation.
The control path must assume that the ground plane is fixed at
`Z=-0.2 m` in the official camera coordinate convention and must no longer
depend on the robot's live D435 mounting transform.

The live D435 transform remains required for RGB-D BEV point-cloud rendering.
This change separates control-path interpretation from real-camera point-cloud
geometry.

## Coordinate Convention

NavDP's official RGB renderer assigns every trajectory point a camera-frame
height of `Z=-0.2 m`. This means that the ground is 0.2 metres below a level
virtual camera. In the existing `base_from_camera` convention, the equivalent
synthetic transform has:

- the standard level optical-to-base rotation;
- zero planar translation;
- camera origin `z=+0.2 m` in the synthetic base frame;
- no pitch, roll, or yaw correction.

The value `-0.2` must not be copied directly into
`base_from_camera[2, 3]`. That entry represents camera height in the destination
frame and therefore has the opposite sign.

## Data Sources

### Retained real-camera data

The client continues subscribing to
`/cam_head/d435/color/camera_info`. The live `CameraInfo.k` matrix is:

- sent to the NavDP server during `navigator_reset`;
- used by the official selected-trajectory pixel projection;
- used by the matching synthetic-camera ray calculation;
- used with the live D435 transform for RGB-D BEV backprojection.

The aligned color-depth input remains unchanged.

### Removed control dependency

The selected-to-guide control transformation will not use:

- live D435 translation;
- live D435 height;
- live D435 pitch, roll, or yaw;
- camera planar offsets when converting the selected path to odom.

The live transform is still captured with each RGB-D frame because the BEV
point cloud needs the real camera pose.

## Control Geometry

For each selected local waypoint `(forward_x, left_y)`:

1. Project it with the live intrinsic matrix and official
   `Z=-0.2 m` formula:

   ```text
   u = fx * (-left_y / forward_x) + cx
   v = image_height - 1 + fy * (0.2 / forward_x) - cy
   ```

2. Convert `(u, v)` back into a ray using the same live intrinsic matrix.
3. Transform the ray with a fixed level optical-to-base rotation whose camera
   origin is `(0, 0, +0.2)`.
4. Intersect the ray with the synthetic base ground plane `z=0`.
5. Convert the resulting base XY path to odom using only the planning-frame
   robot odometry pose.

The existing finite-value, positive-forward, forward-progress, and valid-ray
checks remain fail closed.

Because the official projection and inverse projection use matching geometry,
the yellow guide should closely overlap the raw selected trajectory. Small
differences can remain when the live principal point is not exactly at the
image centre, because the official renderer's vertical formula includes both
the image height and `cy`.

## Components

### Synthetic official transform

Add a pure helper that constructs the fixed level optical-to-base transform
for the official 0.2-metre camera height. The helper owns the sign convention
and prevents planning code from constructing an ambiguous matrix inline.

### Planning path installation

The planning loop will:

1. preserve the server-selected local XY trajectory;
2. transform the raw selected diagnostic to odom using robot odometry only;
3. call the existing reprojection function with the live intrinsic matrix,
   image height, and synthetic official transform;
4. transform the reprojected guide to odom using robot odometry only;
5. install the complete guide into the fixed-horizon MPC as before.

No MPC horizon, weights, sampling, critic, or candidate-selection behavior
changes.

### RGB-D BEV

RGB-D BEV backprojection continues using the live `base_from_camera` stored in
the frame snapshot. The green odometry, red MPC prediction, cyan selected, and
yellow guide layers keep their current meanings.

The cyan and yellow layers are expected to overlap closely after this change:

- cyan: raw selected diffusion under official planar interpretation;
- yellow: the same selected path after official intrinsic/pixel/ray geometry;
- red: MPC prediction from the robot's current state.

## Configuration and Diagnostics

The official control height is fixed at 0.2 metres. Remove the
`--virtual-camera-height` runtime override so a launch command cannot silently
select non-official control geometry.

Plan diagnostics will continue recording:

- live intrinsic-derived projection inputs through the existing plan data;
- raw selected world XY;
- reprojected base and world XY;
- the fixed official height as `virtual_camera_height_m=0.2`;
- the live camera pose for BEV/TF diagnosis;
- reprojection status and rejection reason.

The diagnostic camera pose must not imply that it affected the control path.
Documentation will state that it is retained for real-camera visualization and
TF troubleshooting only.

## Failure Behaviour

- Missing camera intrinsics still prevents planning-frame creation.
- Missing live camera TF still prevents the current combined RGB-D snapshot,
  because BEV backprojection continues to require it; decoupling frame capture
  from BEV TF availability is outside this change.
- Invalid official projection or ground intersection invalidates the candidate.
- A rejected candidate is never partially installed into MPC.
- Existing fail-closed stop behavior remains unchanged.

## Testing

Focused tests will prove:

1. the synthetic official transform represents a level optical camera at
   `+0.2 m`, equivalent to ground `Z=-0.2 m`;
2. changing a supplied live D435 TF height or pitch does not change the
   control guide installed from the same selected trajectory;
3. the live `CameraInfo.k` matrix still reaches NavDP initialization and the
   official projection function;
4. selected and guide odom conversion does not apply live camera XY/yaw;
5. RGB-D BEV backprojection still receives the live frame
   `base_from_camera`;
6. reprojection validation remains fail closed;
7. fixed MPC horizon and full-guide visualization behavior remain unchanged.

Run the focused geometry and client-source tests first, then the complete
image-goal client test suite.

## Non-Goals

- Changing NavDP model inference, critic scoring, or candidate count.
- Changing MPC horizon, costs, control limits, or reference sampling.
- Removing TF acquisition from the RGB-D/BEV pipeline.
- Calibrating a new real-camera height or pitch.
- Adding a runtime switch between official and live control extrinsics.
- Changing depth obstacle geometry or BEV point-cloud placement.
