# MPC Guide Heading Reference Design

## Goal

Make the kinematic MPC naturally produce low linear velocity and high angular
velocity when the installed guide turns sharply.

## Design

Replace the fixed zero-yaw reference generated in `Mpc_controller.solve()` with
heading references derived from the sampled guide:

- Compute each reference heading from the tangent between consecutive sampled
  XY reference points.
- Unwrap tangent headings across `-pi`/`pi`.
- Lift the first heading to the equivalent angle closest to the robot's current
  yaw so the optimizer does not request a full revolution at the wrap boundary.
- Reuse the nearest valid tangent for coincident points and preserve the current
  robot yaw only when all sampled reference points coincide.
- Penalize wrapped heading error in the MPC objective in addition to XY error.

The guide XY sampling, full 24-point input, `update_ref_traj()` warm start,
camera projection, arrival checks, and command deadman remain unchanged.

## Control Parameters

- Restore the deployed NavDP horizon to `N=15`.
- Set `desired_v` to the deployed `v_max`, so reference spacing is dynamically
  feasible at the configured real-robot speed.
- Restore `w_max=0.5 rad/s`.
- Use `Q = diag([10, 10, 5])` and retain the existing
  `R = diag([0.02, 0.15])` control-effort cost.

No separate rotate-first state machine is introduced. Low `v` and high `w`
must arise from the MPC optimization against the guide tangent.

## Verification

- A straight guide aligned with the robot produces forward motion with small
  angular velocity.
- A 90-degree guide tangent produces a much smaller first-step linear velocity
  and a materially larger angular velocity.
- Equivalent headings across the `-pi`/`pi` boundary do not reverse turn
  direction or request a full revolution.
- Updating `ref_traj` preserves the previous MPC warm-start arrays.
- The complete client test suite passes.
