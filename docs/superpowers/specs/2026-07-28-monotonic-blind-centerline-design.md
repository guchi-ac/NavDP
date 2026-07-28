# Monotonic Blind-Zone Centerline Design

## Context

The current `TrajectoryManager` uses one polyline for two different kinds of
state:

1. persistent planned geometry; and
2. the transient MPC requirement that the reference start at the current
   chassis position.

During history advancement it writes
`chassis -> closest projection -> old path tail` back into persistent history.
The chassis-to-projection segment is approximately perpendicular to the
projected path segment. Repeating this operation accumulates centimetre-scale
right-angle turns even when the visible far-field trajectory barely changes.

In `20260728_031428_mpc.jsonl`, the initial `0.246 m` blind region contained
four guides and no turns above 45 degrees. Later, a `0.544 m` blind region
contained 52 guides and 31 turns above 45 degrees. A same-plan control sample
then changed angular command from `+0.5` to `-0.5 rad/s` after about `9 mm` of
chassis translation because the local reference tangent changed by about
61.5 degrees.

## Goals

- Keep persistent planned geometry independent of the current chassis anchor.
- Never persist a temporary chassis-to-projection connector.
- Guarantee that blind guides advance strictly monotonically along centerline
  arc length.
- Make blind-guide spacing comparable to the current diffusion-guide spacing.
- Bound blind-guide count by physical blind length rather than update count.
- Preserve every normalized point of a newly accepted diffusion candidate.
- Keep the existing candidate distance and heading acceptance gates.
- Keep the existing MPC reference-distance sampling and `ref_gap` behavior.

## Non-Goals

- No B-spline, moving average, blend zone, point averaging, or candidate
  fusion.
- No angular acceleration penalty or MPC weight change in this change.
- No requirement that Euclidean distance from the chassis increase at every
  guide. That would reject valid curved paths. Monotonicity is defined by
  centerline arc-length progress.
- No persistent commit horizon or fixed one-metre boundary.

## First-Principles State Separation

`TrajectoryManager` will maintain separate persistent and transient state.

### Persistent state

- `history_centerline`: an odometry-frame polyline representing accepted
  planned geometry. It contains no chassis-to-projection correction created
  during advancement.
- `diffusion_tail`: the exact normalized suffix from the most recently
  accepted diffusion candidate.
- `diffusion_start_arc`: the arc coordinate of `diffusion_tail[0]` on
  `history_centerline`.
- `diffusion_spacing`: the robust typical spacing of the current diffusion
  candidate.
- `last_chassis`: the chassis position from the previous update, used to bound
  forward projection search.

### Transient output

- `active_traj`: a newly constructed MPC source reference. It starts at the
  current chassis, contains uniformly spaced forward blind guides, and ends
  with the exact diffusion suffix.

`active_traj` is never assigned back to `history_centerline`.

## Diffusion Spacing

For a normalized candidate, calculate all consecutive segment lengths and use
their median:

```text
candidate_spacing = median(norm(diff(candidate)))
blind_spacing = max(candidate_spacing, configured_min_spacing)
```

The existing `trajectory_point_spacing` setting becomes the lower safety
bound, not the normal blind sampling interval. With the observed diffusion
spacing near `0.15 m`, a `0.54 m` blind region therefore contains about three
blind guides rather than 52.

When a candidate is rejected or unavailable, retain the last accepted
`diffusion_spacing`.

## Monotonic History Advancement

Advancement uses projection only to measure progress.

1. `history_centerline` already begins at the previous progress boundary.
2. Compute chassis displacement since `last_chassis`.
3. Search for the closest centerline projection only within the forward arc
   window:

   ```text
   0 <= projection_arc <= chassis_displacement + diffusion_spacing
   ```

   The one-spacing allowance handles normal sampling and odometry offset while
   preventing a self-crossing path from jumping arbitrarily far forward.
4. Trim persistent history to:

   ```text
   projection -> unchanged old centerline tail
   ```

5. Never prepend `chassis` to the persistent centerline.
6. Decrease `diffusion_start_arc` by the consumed projection arc.
7. Remove already-passed points from a retained diffusion suffix. This rule
   applies only during fallback; a newly accepted candidate is always
   preserved in full.

Because the stored centerline is trimmed but never re-anchored laterally, its
geometry cannot accumulate perpendicular correction segments.

## Candidate Acceptance and Persistent Update

After advancing history:

1. Normalize the current candidate without resampling it.
2. Project `candidate[0]` onto every forward centerline segment and select the
   minimum-distance projection. If projections are equal within numerical
   tolerance, select the one with the smallest arc coordinate. This preserves
   the geometrically correct join on ordinary paths while making exact
   self-crossing ties deterministic and forward-order preserving.
3. Reject a join whose projected arc is behind current progress or whose
   distance/heading gates fail. An equal-arc zero-blind join remains valid.
4. Construct the proposed persistent centerline as:

   ```text
   advanced historical centerline through join projection
   + optional accepted join connector
   + complete normalized candidate
   ```

5. Do not include the current chassis anchor.
6. Store the complete candidate as `diffusion_tail`.
7. Record its start arc and median spacing.

The join connector is valid planned geometry between two accepted plans. It is
different from the transient chassis-to-projection correction and may persist.

At startup, when no historical geometry exists, the straight
`initial chassis -> candidate[0]` segment is intentional planned geometry and
may seed the initial centerline.

## Active Reference Construction

After advancement and optional candidate acceptance:

1. Treat the trimmed centerline start as arc `s=0`.
2. Do not emit the projection point.
3. Emit blind guides only at:

   ```text
   diffusion_spacing
   2 * diffusion_spacing
   3 * diffusion_spacing
   ...
   ```

   strictly before `diffusion_start_arc`.
4. Require the first distinct reference after the chassis, whether it is a
   blind guide or a diffusion point, to satisfy:

   ```text
   dot(first_reference - chassis, initial_centerline_tangent) > 0
   ```

   Skip non-forward blind samples. A diffusion point coincident with the
   chassis is the zero-blind anchor and is ignored for this check; validate the
   next distinct point. Reject a newly accepted candidate if its first
   distinct diffusion reference fails this check.
5. Append the exact diffusion suffix.
6. Set:

   ```text
   active_traj = chassis + blind_guides + diffusion_tail
   K_blind = len(blind_guides)
   N = len(diffusion_tail)
   prediction_steps = K_blind + N
   ```

If the blind interval is shorter than one diffusion spacing, emit no blind
guide and connect the transient chassis anchor directly to the diffusion first
point.

This construction guarantees strictly increasing centerline arc coordinates
for blind guides. It intentionally does not guarantee monotonically increasing
Euclidean radius on sharp curves.

## Fallback and Exhaustion

- Rejected, invalid, or low-critic candidates do not mutate the accepted
  centerline geometry.
- The manager still advances progress and reconstructs a transient active
  reference from the remaining centerline.
- Retained diffusion points already passed by the chassis are pruned so no
  output guide lies behind current progress.
- If fewer than two distinct forward reference points remain or remaining
  length is below `min_remaining`, report `history_exhausted`.
- Candidate acceptance is transactional: proposed geometry, spacing, and
  diffusion metadata replace persistent state only after all gates pass.

## Diagnostics

Keep the existing blind/diffusion length and count fields, and add:

- `trajectory_diffusion_spacing_m`
- `trajectory_projection_advance_m`
- `trajectory_blind_max_turn_deg`

These fields make point-density regression and tangent instability visible in
future JSONL logs. `trajectory_blind_max_turn_deg` is the maximum heading
change over `chassis + blind_guides + diffusion_tail[0]`, so it covers both the
initial transient seam and all blind segments.

## MPC and Client Integration

No MPC formulation change belongs in this fix.

- Public `N` remains the number of retained diffusion points.
- `prediction_steps` remains `K_blind + N`.
- `ref_gap`, reference arc-distance sampling, weights, and control constraints
  remain unchanged.
- The client keeps rebuilding only when total prediction dimension changes.

The new manager should make `K_blind` change only when physical blind length
crosses a diffusion-spacing boundary, substantially reducing unnecessary MPC
rebuilds and warm-start resets.

## Required Tests

1. Repeated lateral chassis offsets on a straight centerline never add a
   persistent right-angle connector.
2. One hundred advancement/update cycles do not increase blind count for a
   fixed physical blind length.
3. Blind-guide arc coordinates are strictly increasing.
4. The first distinct reference after the chassis has positive dot product
   with the initial centerline tangent.
5. Blind spacing matches the current candidate segment-length median within
   endpoint tolerance.
6. `K_blind` is bounded by physical blind length divided by diffusion spacing,
   independent of update count.
7. A newly accepted diffusion candidate remains exactly equal to the
   `active_traj` suffix.
8. Missing, low-critic, malformed, distant, and heading-discontinuous
   candidates preserve accepted centerline geometry.
9. Self-crossing history cannot project backward or jump beyond the bounded
   forward projection window.
10. Zero-blind, short candidate, and history-exhaustion behavior remain valid.
11. Replay `20260728_031428_mpc.jsonl` and verify that blind count does not
    accumulate, local blind turns do not reproduce the observed 90-degree
    ladder, and manager update time remains below `200 ms`.
12. Existing controller, client, BEV, and compilation regressions remain
    green.

## Acceptance Criteria

- No transient chassis-to-projection connector is stored in persistent state.
- All emitted blind guides have strictly increasing centerline progress.
- Point count is determined by blind length and diffusion spacing, not by the
  number of planning updates.
- The complete newly accepted diffusion candidate is preserved.
- The latest-log reproduction no longer generates the observed centimetre-scale
  90-degree blind-path ladder.
- No controller-weight adjustment is needed to demonstrate the root-cause fix.
