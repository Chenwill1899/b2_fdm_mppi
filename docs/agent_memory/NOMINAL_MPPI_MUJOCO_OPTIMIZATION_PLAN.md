# Nominal MPPI MuJoCo Optimization Plan

Last updated: 2026-05-07

## Context

The current MuJoCo Scout nominal path is functional but too conservative and too sensitive to local reward-map artifacts. The canonical runtime profile should be `configs/mujoco_scout.yaml`; the older `configs/mujoco_test_obstacles*.yaml` profiles are smoke profiles.

Observed issues:

- LiDAR rays hitting flat ground can create circular high-cost bands in `/msg_local_reward.reward_cost`.
- Goals behind the robot are not handled as a distinct motion phase, so MPPI can mix turning, lateral motion, and small reverse commands.
- Obstacle avoidance relies mainly on center-point sampling of the local costmap; it does not account for the Scout footprint.
- Recent nominal commands show average planar command speed around `0.47 m/s`, with very low lateral usage, below the desired stable `1.0 m/s` envelope.

Chosen behavior:

- Behind-goal behavior: rotate toward the target first, then translate forward. Do not make reverse driving the default.
- Speed target: stable navigation within `robot.max_vx=1.0 m/s`; improve average speed by reducing over-conservatism, not by raising the max speed.

## Implementation Scope

- Keep the implementation in py-mppi first. Do not modify Geomapping/MEDIRL in this stage.
- Add local costmap filtering in `b2_fdm_mppi.mujoco_closed_loop.LocalCostmapAdapter`.
- Add footprint-aware local costmap scoring in nominal MPPI controllers.
- Add a rotate-then-translate motion policy in the MuJoCo closed-loop node.
- Update `configs/mujoco_scout.yaml` and make the one-shot launch path use it.

## Key Changes

- Local costmap ground-artifact filter:
  - Use `height`, `roughness`, `cost_map`, and `occupancy` to clear high `reward_cost` cells that look like flat observed ground.
  - Keep raw and filtered cost statistics in the costmap snapshot.
  - Keep obstacle-like cells if roughness, slope, occupancy, or height exceed thresholds.

- Footprint-aware costmap scoring:
  - Sample multiple points around the robot footprint for each predicted state.
  - Use the maximum sampled local cost per state so centerline-only plans cannot graze obstacles.
  - Keep NumPy and Torch nominal cost behavior aligned.

- Rotate-then-translate policy:
  - Add a motion policy that enters rotate mode for large heading error and exits with hysteresis.
  - While rotating, plan to the current position with target yaw toward the goal.
  - Keep `allow_reverse=false` for the Scout profile.

- Parameter cleanup:
  - Tune `mujoco_scout.yaml` for `max_vx=1.0`, moderate acceleration, less over-smoothing, and enough yaw authority.
  - Keep `mujoco_test_obstacles*.yaml` as smoke profiles, not the main one-shot launch target.

## Validation

Targeted tests:

- Costmap artifact filter clears high-cost flat-ground cells and preserves obstacle-like cells.
- Footprint-aware costmap scoring penalizes footprint overlap even when the center path looks clear.
- Rotate-then-translate enters/exits with hysteresis and does not prefer reverse motion.
- `mujoco_scout.yaml` remains the canonical direct local-costmap Scout profile.

Runtime acceptance:

- Flat goal-ahead case reaches within `0.5 m`.
- Behind-goal case rotates first, avoids sustained reverse command below `-0.05 m/s`, then drives forward.
- Obstacle/local-costmap case avoids high-cost obstacle cells without obvious grazing.
- Nominal NumPy mean MPPI time remains below `25 ms`.
- Successful nominal run has mean commanded planar speed above `0.70 m/s` and cruise p80 above `0.85 m/s` where the path is clear.

## Execution Status

Implemented on 2026-05-07:

- Added flat-ground reward-ring filtering in `LocalCostmapAdapter`.
- Added footprint-aware local costmap scoring for NumPy and Torch nominal MPPI.
- Added rotate-then-translate motion policy for large behind-goal heading error.
- Updated `configs/mujoco_scout.yaml` for the canonical nominal MuJoCo Scout profile.
- Updated `launch_mppi_sim.sh` and `README.md` to use `configs/mujoco_scout.yaml`.
- Added focused tests for artifact filtering, motion policy hysteresis, footprint scoring, and NumPy/Torch cost alignment.

Verification:

- `/usr/bin/python3 -m pytest tests/test_mujoco_closed_loop.py tests/test_mppi_omni_numpy.py tests/test_mppi_omni_torch.py tests/test_config.py -q`
  - Result: `106 passed in 3.54s`.
- `git diff --check`
  - Result: passed.

Notion sync:

- Created task-board entry: `MuJoCo nominal MPPI local-costmap 优化收口`.
- Created experiment entry: `2026-05-07 nominal_numpy MuJoCo optimization unit validation`.
- Created decision entry: `MuJoCo nominal MPPI 先在 py-mppi 侧过滤与调参`.
- Updated main `fdm_mppi` project page focus text with the 2026-05-07 completion note and next validation step.
