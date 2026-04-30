# Task Board

Last updated: 2026-04-30

## Current Stage

Stage 1.5: B2 omnidirectional SE(2) nominal kinodynamic rollout.

## Stage 0 Acceptance Criteria

- No fatal error.
- Final distance to goal `< 0.4 m`.
- Average MPPI computation time `< 20 ms`.
- Summary, CSV, and PNG outputs are saved.
- `animation.gif` is saved for stage inspection when animation is enabled.
- Test reports are saved under `results/test_reports/<timestamp>/` for user verification.
- Animation failure must not break the main simulation flow.

## Immediate Tasks

| ID | Priority | Status | Task | Notes |
| --- | --- | --- | --- | --- |
| S0-001 | P0 | done | Save GIF and make animation failure non-fatal | Default animation enabled for `animation.gif`; runner catches animation exceptions and continues with a warning. Verified by pytest. |
| S0-002 | P0 | done | Add baseline summary metrics | `test_summary.yaml` includes success, final distance, path length, arrival/run time, mean/max MPPI time. Verified by pytest. |
| S0-003 | P0 | done | Create short-goal baseline config | Added `config/fdm_mppi_baseline_short.yaml`: target `[3.0, 3.0]`, 400 steps, 2.0 s horizon, 1024 samples, GIF enabled. Verified by pytest. |
| S0-004 | P1 | done | Run baseline with real controller | Real PyCUDA MPPI ran and saved GIF under `results/sim_results/2026-04-29_22-16-38/`; acceptance failed with `final_distance=1.2289 m`. |
| S0-005 | P1 | todo | Save trajectory/control/time plots and GIF reliably | Existing plot hooks are present; verify CSV, PNG, and `animation.gif` outputs. GIF y-axis is fixed to `[-10, 10]` for baseline inspection. |
| S0-006 | P0 | todo | Tune short-goal baseline to reach target | Need reduce `final_distance` from `1.2289 m` to `<0.4 m` while keeping mean MPPI time `<20 ms`. |
| S0-007 | P0 | done | Add straight-obstacle Stage 0 scene | Updated target `[18, 0]`, static obstacles `[6, 0.5]` and `[12, -1]`, fixed map limits `x=[0, 20], y=[-10, 10]`. Obstacles verified stationary. |
| S0-008 | P0 | todo | Tune harder double-obstacle scene | Current run failed with `final_distance=7.5119 m`; min clearance `0.2870 m`; mean MPPI `1.2541 ms`. Need reach target and keep clearance >= 0.3 m. |
| S1-001 | P0 | done | Add B2 omnidirectional SE(2) nominal model | Added `core/omni_b2.py` with state `[x,y,theta,vx,vy,wz]`, control `[vx,vy,wz]`, and limits `1.5/0.5/1.0`. Verified by pytest. |
| S1-002 | P0 | done | Add B2 omni nominal config | Added `config/b2_omni_nominal.yaml`; config validation now supports state/control dimensions from config. |
| S1-003 | P0 | done | Add NumPy omni MPPI rollout/controller | Added `controllers/mppi_omni_numpy.py`; supports 3D controls, config factory, obstacle cost, and closed-loop smoke test. Verified by pytest. |
| S1-004 | P0 | done | Run and tune B2 omni baseline | Added omni runner/logger, restored candidate and optimal rollout GIF drawing, tuned `obstacle_weight=800`, `safety_dist=0.4`, `smooth_weight=1.0`. Verified final run `results/sim_results/2026-04-29_23-27-26/`: success, final distance `0.3634 m`, mean MPPI `6.3741 ms`, GIF saved. |
| S1-005 | P0 | done | Add simple timestamp result suffix | Formal omni run now writes `results/sim_results/b2_omni_nominal_<timestamp>/`, avoiding overwrite while keeping runs easy to identify. Verified by pytest and real GIF run. |
| S1-006 | P0 | done | Add CUDA backend for B2 omni MPPI | Added `MppiOmniCuda` with PyCUDA rollout/cost kernel and `mppi.backend: cuda`. Verified by pytest and real run `results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/`: success, final distance `0.3754 m`, mean MPPI `4.9297 ms`. |
| S1-007 | P0 | done | Add CBF cost to CUDA omni MPPI | Added discrete CBF penalty term using `cbf.dcbf_alpha` and `mppi.cbf_weight`; current config uses `cbf_weight=500`. Real run min clearance `0.4296 m`. |
| S1-008 | P0 | done | Integrate and tune RCBF-style CUDA barrier | CUDA omni MPPI now supports `cbf.type` 1/2/3 barrier modes from the old project. Tuned config: `num_trajectories=1024`, `minimum_distance=0.45`, `cbf.type=1`, `cbf_weight=500`. Verified run `results/sim_results/b2_omni_nominal_2026-04-30_14-14-06/`: success, final distance `0.3411 m`, mean MPPI `4.5400 ms`, max `11.8539 ms`, min clearance `0.4715 m`. |
| S1.5-001 | P0 | done | Refine static-obstacle nominal planner smoothness | Default RCBF disabled for the static-obstacle baseline (`cbf.enabled=false`, `cbf.type=0`, `cbf_weight=0`). Added control smoothness/jerk/variance metrics and executed-control low-pass filtering. Formal run `results/sim_results/b2_omni_nominal_2026-04-30_14-52-40/`: success, final distance `0.3399 m`, min clearance `0.4645 m`, mean MPPI `5.1127 ms`, control smoothness `0.01228`, control jerk `0.02026`. |
| S1.5-002 | P0 | done | Add kinodynamic constraints to B2 nominal rollout | Added rollout-internal velocity lag, acceleration limits, lateral/yaw/accel costs, and summary metrics. Verified run `results/sim_results/b2_omni_nominal_2026-04-30_15-46-59/`: success, final distance `0.3212 m`, min clearance `0.4854 m`, mean MPPI `4.7268 ms`, trajectory y-span `1.9814 m`, executed max delta `[0.032, 0.020, 0.048]`. |
| S1.5-003 | P0 | done | Keep all assistant work on dev | User rule recorded: modify only on `dev`, push `dev`, user merges. Accidental feature branch work migrated back to `dev`. |

## Later Stages

| Stage | Status | Goal |
| --- | --- | --- |
| 1.5 | in_progress | B2 omnidirectional SE(2) nominal kinodynamic rollout. |
| 2 | pending | Oracle residual world. |
| 3 | pending | Unified evaluation system. |
| 4 | pending | Oracle dataset generation. |
| 5 | pending | Residual velocity FDM training. |
| 6 | pending | Learned FDM-MPPI integration. |
| 7 | pending | Paper-ready experiments and figures. |
