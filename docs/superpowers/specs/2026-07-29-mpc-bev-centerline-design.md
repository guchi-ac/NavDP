# MPC BEV Centerline Design

## Goal

Add a subtle longitudinal centreline to `mpc_rgb_bev` so the operator can
quickly judge whether selected, guide, MPC, and measured paths lie to the left
or right of the robot's forward axis.

## Rendering

- The line represents robot-frame `y=0`.
- It spans the complete configured BEV range from `x=-rear_m` to
  `x=+forward_m`.
- It is a one-pixel dark-grey solid line.
- It is rendered after the RGB-D point cloud and before all path layers.
- Existing green actual, red MPC, cyan selected, yellow guide, and white robot
  layers therefore remain visually dominant.
- The line has no legend entry because it is a spatial reference rather than a
  data series.
- If current odometry is unavailable, the line is still shown because it is
  defined directly in the robot/BEV frame.

## Scope

Modify only the MPC RGB BEV renderer, its rendering tests, and the Chinese
operator documentation. Do not change projection, trajectory selection, MPC,
freshness gating, video encoding, or RGB-D geometry.

## Testing

Add a rendering test that uses an empty point cloud and verifies dark-grey
pixels along the configured `y=0` column away from the white robot marker.
Existing layer tests must continue proving that paths and the robot marker
draw over lower layers.
