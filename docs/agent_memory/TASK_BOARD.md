# Task Board

Last updated: 2026-05-01

## Current Stage

Stage 5: learned residual FDM NumPy closed-loop benchmark.

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
| S1.5-004 | P0 | done | Finalize Pure SE2 and Kinodynamic baselines | Added `config/b2_omni_pure_se2.yaml` and `config/b2_omni_kinodynamic.yaml`; added far-field static obstacle potential, sampling coverage summary metrics, and kinodynamic animation rollout. Verified Pure run `results/sim_results/b2_omni_pure_se2_2026-04-30_16-26-22/` and Kinodynamic run `results/sim_results/b2_omni_kinodynamic_2026-04-30_16-27-08/`. |
| S2.6-001 | P0 | done | Prepare fixed-map random-task oracle dataset config | `config/b2_omni_oracle_random100_dataset.yaml` uses `scenario.random_seed: auto` for single-run random start/goal, fixed obstacle/terrain/oracle seeds, PNG enabled, GIF disabled. Stage 3 seed modes documented in `docs/agent_memory/STAGE3_SEEDING.md`. |
| S3.5-001 | P0 | done | Add parallel oracle episode generation | `tools/generate_oracle_episodes.py` supports `--num-workers`; manifest stays sorted by `episode_id`; single episode failures do not stop the run. |
| S3.5-002 | P0 | done | Harden parallel collection outputs | `collect_oracle_episode.py` writes raw results to `raw_results/episode_XXXXXX`; `trajectory.csv` includes final state; dataset builder resolves relative manifest paths; validator reports missing fields and manifest/data mismatches. |
| S4-001 | P0 | done | Start residual FDM training baseline | Added `tools/train_residual_fdm.py`, requirements docs, and tests. Short run on `datasets/oracle_stage3_splits`: val_mse `1.093e-05`, test_mse `1.104e-05`, zero residual val/test MSE `6.823e-04` / `6.438e-04`. |
| S4-002 | P0 | done | Add TensorBoard training visualization | Residual FDM training writes scalar curves and val residual diagnostic figures under `tensorboard/` in the run output. |
| S4-003 | P0 | done | Add open-loop learned FDM rollout evaluation | Added `tools/evaluate_residual_fdm_rollout.py` to replay nominal vs learned FDM against oracle ground truth and render `rollout_compare.gif` with terrain risk. Standard eval scene is `config/b2_omni_oracle.yaml` with `robot.safety_dist=0.25`; seed123 run saved under `results/fdm_rollout_eval/stage4_mlp_seed123_b2_omni_oracle_seed123/`: learned ADE `0.04871 m` vs nominal ADE `0.52125 m`, learned FDE `0.14396 m` vs nominal FDE `0.91905 m`, residual MSE improvement `98.94%`. Metrics now include reproducibility metadata, explicit checkpoint/normalization paths, no stale GIF reuse under `--no-gif`, and ADE/FDE at `1s/2s/4s`. |
| S4-004 | P0 | done | Harden residual FDM training baseline | `tools/train_residual_fdm.py` now writes reproducibility metadata, per-axis MSE/RMSE/improvement fields, TensorBoard best-val scalars, and both `best_model.pt` plus final `model.pt`. Hardened seed123 run on `datasets/oracle_stage4_splits` saved under `results/fdm_baselines/stage4_mlp_seed123_hardened/`: val/test MSE `1.000e-05` / `1.005e-05`, zero residual val/test MSE `6.535e-04` / `6.289e-04`, relative improvement `98.47%` / `98.40%`, best epoch `46`, final epoch `50`. |
| S4-005 | P0 | done | Close Stage 4 benchmark and OOD protocol | Added OOD obstacle/terrain dataset configs, `tools/evaluate_residual_fdm_dataset.py`, and `docs/agent_memory/STAGE4_PROTOCOL.md`. Multi-seed `123/456/789` mean test MSE `1.015e-05` with mean test reduction `98.386%`. Best checkpoint standard open-loop learned ADE/FDE `0.04844/0.08230` vs nominal `0.52125/0.91905`. OOD obstacle/terrain one-step test improvement `57.20x` / `59.64x`; OOD rollout learned ADE/FDE `0.02070/0.04631` and `0.02465/0.05385`, both better than nominal. |
| S5-001 | P0 | done | Add learned residual dynamics wrapper and NumPy smoke path | Added shared residual FDM model module, `LearnedResidualDynamics`, `LearnedFdmMppiOmniNumpy`, FDM config/CLI overrides, and `docs/agent_memory/STAGE5_PROTOCOL.md`. Standard scene smoke passed stability threshold: nominal final distance `0.3282 m`, learned final distance `0.3430 m` under threshold `0.4938 m`; learned reached goal and failed=false. Runtime is high (`mean_mppi_time_ms=1143.5`) and must be optimized before real-time claims. |
| S5-002 | P0 | done | Add Stage 5 closed-loop benchmark runner | Added `tools/benchmark_learned_fdm_mppi.py` and `docs/agent_memory/STAGE5_BENCHMARK.md` for paired nominal vs learned NumPy oracle benchmarks. The tool writes `stage5_benchmark_summary.json` with metadata, per-run metrics, per-controller aggregates, and learned-minus-nominal paired deltas. Full ID/OOD benchmark results are deferred to PR #19. |
| S5-003 | P0 | done | Add Torch CUDA learned rollout and run Stage 5-B benchmark | Added `LearnedFdmMppiOmniTorch` and enabled `fdm.enabled=true` with `mppi.backend=cuda`. Ran standard plus 20-episode ID/OOD obstacle/OOD terrain benchmarks. Learned improves the standard scene and reaches 100% success in ID/OOD, but does not stably beat nominal on random-task final distance/steps/clearance. Torch CUDA learned runtime is practical (`~27 ms` standard, `~51-54 ms` ID/OOD) but still slower than nominal CUDA. |
| S5-004 | P0 | done | Standardize Stage 5 closed-loop visual eval | Added `tools/visualize_stage5_closed_loop.py` and `docs/agent_memory/STAGE5_VISUAL_EVAL.md`. The tool runs paired nominal/learned oracle closed-loop simulations with plots and GIFs enabled, then writes `closed_loop_nominal_vs_learned.png`, metric CSV/JSON, and `stage5_visual_eval_summary.json`. |

## Later Stages

| Stage | Status | Goal |
| --- | --- | --- |
| 1.5 | done | B2 omnidirectional SE(2) nominal kinodynamic rollout. |
| 2 | done | Oracle residual world. |
| 3.5 | done | Parallel Oracle Dataset Generation with explicit episode seed mapping. |
| 4 | done | Residual velocity FDM training baseline and open-loop/OOD validation. |
| 5 | in_progress | Learned FDM-MPPI NumPy integration, closed-loop benchmark, and runtime profiling. |
| 6 | pending | Unified evaluation system. |
| 7 | pending | Paper-ready experiments and figures. |
