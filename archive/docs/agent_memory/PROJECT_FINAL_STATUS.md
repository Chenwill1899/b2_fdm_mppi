# Project Final Status: Risk-Aware Learned-FDM-MPPI Numerical Package

Date: 2026-05-03

## Scope

This is the final convergence status for the current numerical simulation package. It covers Stage 5-D, Stage 5-E, and Stage 6 runtime profiling after PR #28 was merged into `fdm`.

This package does not add a new model architecture, does not add new maps, and does not include MuJoCo or real-robot closed-loop deployment.

## Final Claim

The repository now supports a reproducible, paper-ready numerical evaluation of risk-aware learned-FDM-MPPI:

- explicit terrain-risk MPPI cost is implemented and evaluated;
- same-backend Torch nominal-vs-learned risk-aware ablations are available;
- map-level conclusions are documented with paired statistics and figures;
- runtime profiling is closed as a bounded limitation rather than a real-time claim;
- fixed two-obstacle visual evidence is packaged for continuity with earlier figures.

## What Is Supported

### Risk-aware planning claim

The planner can explicitly penalize terrain risk through MPPI cost terms:

- `terrain_risk_weight`
- `terrain_risk_power`
- `terrain_risk_threshold`
- `terrain_risk_mode`

This supports the statement that the planner is risk-aware in the numerical oracle simulator.

### Learned-vs-nominal claim

Learned-FDM-MPPI should be described map-by-map, not as a global winner:

| Map | Status | Supported wording |
| --- | --- | --- |
| `low_friction_patch` | strong | Learned risk-aware MPPI reduces cumulative risk, excess risk, and exposure versus same-backend nominal risk-aware MPPI. |
| `safe_corridor` | supporting | Learned reduces cumulative risk and steps, while excess/exposure are not clean wins. |
| `risk_band` | limitation | Stress test; learned reduces cumulative risk but does not clearly improve all risk exposure metrics. |
| `two_obstacle_standard` | visual continuity | Fixed-scene risk-map trajectory evidence and 50-episode summary are available; this is not the strongest risk-aware map. |

### Runtime claim

Stage 6 supports a reproducible runtime profiling claim:

- fixed-seed same-backend Torch 2x2 runtime matrix is available;
- runtime profiler forces fixed `compute_control()` counts by default;
- closeout `profile_call_consistency.consistent=true`;
- learned risk-on remains about `+26.02 ms` slower than nominal risk-on in the forced-step closeout profile;
- the dominant gap is learned rollout, not terrain-risk cost, candidate sampling, or distribution update.

This is not a real-time equivalence claim.

## What Is Not Supported

Do not claim:

- learned-FDM-MPPI is globally better than nominal MPPI on every map and metric;
- risk-aware learned-FDM-MPPI is real-time equivalent to nominal Torch MPPI;
- the current implementation is ready for direct real-robot deployment;
- closed-loop MuJoCo or hardware results exist in this package;
- `risk_band` is a clean success case;
- fixed two-obstacle figures are ground truth comparisons.

Closed-loop visual figures are controller executions in the oracle world. They should not be labeled `GT`.

## Tracked Artifacts

Risk-aware result package:

- `docs/agent_memory/STAGE5_E_RISK_AWARE_PROTOCOL.md`
- `docs/agent_memory/STAGE5_E_RISK_AWARE_RESULTS.md`
- `docs/agent_memory/NATURE_FIGURE_STYLE.md`
- `figures/stage5_e/`
- `tables/stage5_e/`

Runtime package:

- `docs/agent_memory/STAGE6_RUNTIME_PROFILE.md`
- `docs/agent_memory/STAGE6_RUNTIME_CLOSURE.md`
- `figures/stage6/`
- `tables/stage6/`

Final convergence package:

- `docs/agent_memory/PROJECT_FINAL_STATUS.md`
- `docs/agent_memory/REAL_ROBOT_READINESS.md`
- `docs/agent_memory/REPRODUCIBILITY_COMMANDS.md`

Raw run outputs remain under `results/` and are not tracked.

## Final Numerical Snapshot

Risk-aware 50-episode same-backend Torch result, learned minus nominal, risk-on official weight:

| Map | Weight | Final delta | Cumulative risk delta | Excess risk delta | Exposure delta | Runtime delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low_friction_patch | 10 | -0.0127 | -9.1263 | -3.7318 | -0.0204 | +21.57 ms |
| safe_corridor | 0.5 | -0.0019 | -2.2375 | +0.0040 | +0.0033 | +27.94 ms |
| risk_band | 5 | -0.0002 | -3.3511 | -0.3686 | +0.0005 | +21.08 ms |
| two_obstacle_standard | 3 | -0.0071 | -7.6334 | -2.7628 | +0.0139 | +17.65 ms |

Runtime closeout, forced 10 control calls per run:

| Case | calls/run | mean_mppi_time_ms | rollout_total_ms |
| --- | ---: | ---: | ---: |
| nominal_risk_on | 10 | 13.91 | 8.54 |
| learned_risk_on | 10 | 39.93 | 34.98 |
| learned risk-on - nominal risk-on | 10 | +26.02 | +26.44 |

## Readiness

The package is ready for:

- paper/report figures and tables;
- numerical reproducibility review;
- algorithmic discussion of map-level risk-aware behavior;
- planning a shadow-mode adapter stage.

The package is not ready for:

- direct real-robot command publication;
- safety-critical deployment;
- real-time learned controller claims;
- MuJoCo or hardware closed-loop claims.

The next stage should be shadow-mode only, with the minimum entry conditions in `docs/agent_memory/REAL_ROBOT_READINESS.md`.
