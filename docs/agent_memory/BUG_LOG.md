# Bug Log

Last updated: 2026-04-30

## Known Risks

| ID | Status | Symptom | Likely Cause | Next Action |
| --- | --- | --- | --- | --- |
| B-001 | fixed | Animation may fail or block run completion | animation writer unavailable; runner previously let animation exceptions propagate | Default animation enabled to save `animation.gif`; runner catches animation exceptions and emits `RuntimeWarning`. Verified by `python3 -m pytest -q` on 2026-04-29. |
| B-002 | open | Real MPPI controller may fail to import/run | PyCUDA/NVIDIA runtime unavailable | Keep fake-controller tests as CPU smoke coverage; verify GPU separately. |
| B-003 | fixed | Summary lacked Stage 0 acceptance metrics | Previous `test_summary.yaml` only stored init, goal, steps, failed, average time | Added success, final distance, path length, arrival/run time, and mean/max MPPI time. Verified by pytest on 2026-04-29. |
| B-004 | fixed | `animation.gif` y-axis too short for baseline inspection | Animation used `plt.ylim(-1, targets[1] + 1)`, which becomes `[-1, 1]` for target y=0 | Added `animation_axis_limits()` and set animation y-axis to `[-10, 10]`. Verified by pytest on 2026-04-29. |
| B-005 | open | Short-goal real MPPI baseline does not reach target | First short-goal run overshot and ended at `final_distance=1.2289 m` after 400 steps | Tune Stage 0 config: increase goal attraction, adjust heading/yaw handling, control costs, and sampling noise one small change at a time. |
| B-006 | fixed | Straight-obstacle scene passed goal but got too close to obstacle | Single-obstacle run reached target but min clearance was `0.2819 m`, below `safety_dist=0.3 m` | Double-obstacle scene with obstacles at `[6, 1]` and `[12, 1.5]` reached min clearance `0.3174 m`. Verified on 2026-04-29. |
| B-007 | fixed | Static obstacle moved toward robot despite zero velocity in config | `Obstacle.update_cur_state_virtual()` decreased obstacle `dx` every step and runner called it even when `static_enabled: true` | Runner now skips virtual obstacle state updates when `obstacles.static_enabled` is true. Verified by pytest and obs CSV on 2026-04-29. |
| B-008 | open | Harder double-obstacle scene does not reach goal | Obstacles `[6, 0.5]` and `[12, -1]` create a tighter path; first run stopped near `[10.54, -0.93]` with `final_distance=7.5119 m` | Tune MPPI parameters for this scene without breaking static obstacle behavior or GIF outputs. |
| B-009 | fixed | Omni runner GIF did not show MPPI sampled/candidate paths | New omni runner used a local animation implementation that only drew executed trajectory | Added per-frame sampled rollout drawing and optimized rollout drawing in `omni_runner.py`. Verified by `test_omni_runner_draws_sampled_and_optimized_rollouts` and saved GIF `results/sim_results/2026-04-29_23-27-26/animation.gif`. |
| B-010 | fixed | Omni trajectory appeared jagged in GIF | Omni MPPI cost penalized control magnitude but not control jumps, especially first control jump from the previous executed command | Added `smooth_weight` and previous-control smoothness cost. Tuned `smooth_weight=1.0`; final run reached target with `final_distance=0.3634 m`, `mean_mppi_time_ms=6.3741`, `min_obstacle_clearance=0.3878`. |
| B-011 | fixed | Formal result checks were overwritten by one fixed latest folder | `results.run_name` with `overwrite: true` reused `b2_omni_nominal_latest` every run | Added `results.timestamp_suffix`; `config/b2_omni_nominal.yaml` now writes to `results/sim_results/b2_omni_nominal_<timestamp>/`. Verified by pytest and real GIF run on 2026-04-30. |
| B-012 | open | CUDA CBF is not yet full soft/slack RCBF | Current omni CUDA controller adds a discrete CBF cost penalty, not the old slack-variable RCBF formulation | Keep the implementation labeled as CBF cost; add soft/slack RCBF later only after CUDA omni baseline remains stable. |

## Fixed Bugs

- B-001: animation failure is non-fatal after `runner._plot_results()` wraps `utils.animate_simulation()`.
- B-003: summary metrics are added to `test_summary.yaml`.
- B-004: baseline animation y-axis is widened to `[-10, 10]`.
- B-006: double static obstacle scene clears `safety_dist=0.3 m`.
- B-009: omni GIF now includes sampled and optimized rollout paths.
- B-010: omni MPPI now includes smoothness cost for adjacent controls and first-step control jump.
- B-011: formal B2 omni nominal result output now uses a named timestamp suffix instead of overwriting one fixed latest directory.
