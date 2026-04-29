# Bug Log

Last updated: 2026-04-29

## Known Risks

| ID | Status | Symptom | Likely Cause | Next Action |
| --- | --- | --- | --- | --- |
| B-001 | fixed | Animation may fail or block run completion | animation writer unavailable; runner previously let animation exceptions propagate | Default animation enabled to save `animation.gif`; runner catches animation exceptions and emits `RuntimeWarning`. Verified by `python3 -m pytest -q` on 2026-04-29. |
| B-002 | open | Real MPPI controller may fail to import/run | PyCUDA/NVIDIA runtime unavailable | Keep fake-controller tests as CPU smoke coverage; verify GPU separately. |
| B-003 | fixed | Summary lacked Stage 0 acceptance metrics | Previous `test_summary.yaml` only stored init, goal, steps, failed, average time | Added success, final distance, path length, arrival/run time, and mean/max MPPI time. Verified by pytest on 2026-04-29. |
| B-004 | fixed | `animation.gif` y-axis too short for baseline inspection | Animation used `plt.ylim(-1, targets[1] + 1)`, which becomes `[-1, 1]` for target y=0 | Added `animation_axis_limits()` and set animation y-axis to `[-10, 10]`. Verified by pytest on 2026-04-29. |
| B-005 | open | Short-goal real MPPI baseline does not reach target | First short-goal run overshot and ended at `final_distance=1.2289 m` after 400 steps | Tune Stage 0 config: increase goal attraction, adjust heading/yaw handling, control costs, and sampling noise one small change at a time. |
| B-006 | open | Straight-obstacle scene passes goal but gets too close to obstacle | Real MPPI run reached target but min clearance was `0.2819 m`, below `safety_dist=0.3 m`; run printed a few `clash!!!` warnings | Increase obstacle/CBF safety effect or tune control/path cost while preserving final distance and compute-time criteria. |
| B-007 | fixed | Static obstacle moved toward robot despite zero velocity in config | `Obstacle.update_cur_state_virtual()` decreased obstacle `dx` every step and runner called it even when `static_enabled: true` | Runner now skips virtual obstacle state updates when `obstacles.static_enabled` is true. Verified by pytest and obs CSV on 2026-04-29. |

## Fixed Bugs

- B-001: animation failure is non-fatal after `runner._plot_results()` wraps `utils.animate_simulation()`.
- B-003: summary metrics are added to `test_summary.yaml`.
- B-004: baseline animation y-axis is widened to `[-10, 10]`.
