# Task Board

Last updated: 2026-04-29

## Current Stage

Stage 0: Baseline tuning and stabilization.

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

## Later Stages

| Stage | Status | Goal |
| --- | --- | --- |
| 1 | pending | B2 omnidirectional SE(2) nominal model. |
| 2 | pending | Oracle residual world. |
| 3 | pending | Unified evaluation system. |
| 4 | pending | Oracle dataset generation. |
| 5 | pending | Residual velocity FDM training. |
| 6 | pending | Learned FDM-MPPI integration. |
| 7 | pending | Paper-ready experiments and figures. |
