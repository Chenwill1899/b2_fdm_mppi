# Stage 5-B Closed-loop Benchmark Protocol

Last updated: 2026-05-01

## Goal

Stage 5-B tests whether the Stage 4 MLP residual FDM improves closed-loop MPPI behavior in the oracle world. This is the first benchmark layer after the PR #17 NumPy smoke path.

This protocol is still Stage 5 validation work. It does not claim full Stage 5 completion. PR #19 adds a Torch CUDA learned-FDM rollout backend so ID/OOD closed-loop benchmark runs are practical, but the benchmark result does not yet show learned-FDM-MPPI is stably better than nominal MPPI on random tasks.

## Benchmark Tool

Tool:

```text
tools/benchmark_learned_fdm_mppi.py
```

Default standard-scene CUDA command:

```bash
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --scenario-name standard \
  --output results/stage5_benchmark/standard_seed123_cuda \
  --episodes 1 \
  --base-seed 123 \
  --backend cuda \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-checkpoint best_model.pt \
  --fdm-normalization normalization.npz \
  --fdm-device cuda
```

The tool runs paired `Nominal-MPPI` and `Learned-FDM-MPPI` simulations in `simulation.world_mode: oracle`. It supports `numpy` and `cuda` backend selection, disables plots and animation by default, and writes each run under an episode-scoped directory. With `fdm.enabled=true` and `mppi.backend: cuda`, the learned controller uses the Torch CUDA rollout backend, not the legacy PyCUDA nominal kernel.

## Output Schema

Main output:

```text
stage5_benchmark_summary.json
```

Top-level sections:

```text
metadata
runs
aggregates
paired_deltas
```

`metadata` records the command, argv, git SHA/branch/dirty flag, config, backend, controllers, episode count, seeds, output directory, and learned FDM artifact paths.

Each `runs` entry records:

```text
scenario
controller
episode_id
seed
results_path
success
failed
final_distance
steps
arrival_time
path_length
min_obstacle_clearance
mean_terrain_risk
mean_cmd_real_error
mean_residual_norm
control_smoothness
control_jerk
mean_mppi_time_ms
max_mppi_time_ms
```

`aggregates` reports per-controller counts, success rate, and mean/std for numeric metrics. `paired_deltas` reports learned-minus-nominal deltas for matched `scenario + episode_id + seed` pairs.

## Stage 5-B Scenarios

PR #19 benchmark commands:

```bash
# Standard scene
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --scenario-name standard \
  --output results/stage5_benchmark/standard_seed123_cuda \
  --episodes 1 \
  --base-seed 123 \
  --backend cuda \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-device cuda

# ID random-task benchmark
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --scenario-name id_random_tasks \
  --output results/stage5_benchmark/id_random_tasks_seed123_cuda \
  --episodes 20 \
  --base-seed 123 \
  --backend cuda \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-device cuda

# OOD obstacle benchmark
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle_random100_dataset_ood_obstacle.yaml \
  --scenario-name ood_obstacle \
  --output results/stage5_benchmark/ood_obstacle_seed123_cuda \
  --episodes 20 \
  --base-seed 123 \
  --backend cuda \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-device cuda

# OOD terrain benchmark
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle_random100_dataset_ood_terrain.yaml \
  --scenario-name ood_terrain \
  --output results/stage5_benchmark/ood_terrain_seed123_cuda \
  --episodes 20 \
  --base-seed 123 \
  --backend cuda \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-device cuda
```

Use `episodes=20` for the first PR #19 pass. Expand to `50` only after the 20-episode run is stable and the runtime is acceptable for offline analysis.

## Metrics And Gates

Core metrics:

```text
success_rate
final_distance
steps
arrival_time
path_length
min_obstacle_clearance
mean_terrain_risk
mean_cmd_real_error
mean_residual_norm
control_smoothness
control_jerk
mean_mppi_time_ms
max_mppi_time_ms
```

Primary gate metrics:

```text
success_rate
final_distance
min_obstacle_clearance
mean_mppi_time_ms
```

Learned-FDM-MPPI should not be called a closed-loop improvement unless it is at least comparable on safety and success while improving final distance or other task-quality metrics.

## PR #19 Results

All runs use `backend: cuda`, `fdm.device: cuda`, and `best_model.pt`.

| Scenario | Episodes | Controller | Success | Final Dist Mean | Steps Mean | Clearance Mean | Terrain Risk Mean | Smoothness Mean | Jerk Mean | Mean MPPI ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 1 | nominal | 1.00 | 0.3461 | 225.0 | 0.1198 | 0.4289 | 0.003071 | 0.003512 | 6.59 |
| standard | 1 | learned | 1.00 | 0.3371 | 214.0 | 0.2024 | 0.4140 | 0.002760 | 0.002405 | 27.03 |
| ID random | 20 | nominal | 1.00 | 0.6795 | 137.1 | 3.5621 | 0.3285 | 0.003186 | 0.002828 | 6.41 |
| ID random | 20 | learned | 1.00 | 0.6930 | 155.8 | 3.5254 | 0.3121 | 0.002782 | 0.002014 | 50.75 |
| OOD obstacle | 20 | nominal | 1.00 | 0.6768 | 142.4 | 2.7875 | 0.3269 | 0.003156 | 0.002765 | 7.19 |
| OOD obstacle | 20 | learned | 1.00 | 0.6878 | 175.1 | 2.8150 | 0.3133 | 0.002764 | 0.002156 | 53.85 |
| OOD terrain | 20 | nominal | 1.00 | 0.6848 | 141.9 | 3.5747 | 0.3633 | 0.003084 | 0.002728 | 6.43 |
| OOD terrain | 20 | learned | 1.00 | 0.6900 | 153.7 | 3.5101 | 0.3409 | 0.002704 | 0.001871 | 51.39 |

Learned-minus-nominal paired deltas:

| Scenario | Final Dist Delta | Steps Delta | Clearance Delta | Terrain Risk Delta | Smoothness Delta | Jerk Delta | Mean MPPI Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | -0.0090 | -11.0 | +0.0826 | -0.0149 | -0.000311 | -0.001107 | +20.44 ms |
| ID random | +0.0135 | +18.7 | -0.0366 | -0.0164 | -0.000404 | -0.000814 | +44.34 ms |
| OOD obstacle | +0.0109 | +32.8 | +0.0276 | -0.0136 | -0.000392 | -0.000609 | +46.67 ms |
| OOD terrain | +0.0051 | +11.8 | -0.0645 | -0.0224 | -0.000381 | -0.000857 | +44.96 ms |

Conclusion:

- CUDA learned rollout makes Stage 5-B benchmark practical: standard learned runtime drops from roughly `1203 ms/step` in the NumPy reference path to `27 ms/step`, and random-task learned runs are around `51-54 ms/step`.
- Learned-FDM-MPPI improves the single standard scene on final distance, steps, clearance, terrain risk, and smoothness.
- On ID/OOD random tasks, learned-FDM-MPPI reaches 100% success but does not stably outperform nominal MPPI on final distance, steps, or clearance. It consistently reduces terrain risk, command-real error, residual norm, smoothness, and jerk.
- Current MLP residual FDM should be treated as a valid closed-loop baseline, not a final improvement claim. The next step is Stage 5-C profiling/tuning and likely rollout/cost calibration before considering history-conditioned FDM.

## Runtime Boundary

The PR #17 smoke measured learned-FDM NumPy MPPI at roughly `1143 ms` mean per step versus nominal NumPy at roughly `13.5 ms`. PR #19 adds a Torch CUDA learned rollout path that brings standard learned mean runtime to roughly `27 ms/step` and ID/OOD learned mean runtime to roughly `51-54 ms/step`.

Stage 5-C should profile before optimizing:

```text
terrain feature/risk
FDM Torch inference
Python horizon loop
cost evaluation
array copy / device transfer
```

Do not replace the MLP model before using the Stage 5-B evidence to tune closed-loop rollout/cost calibration and profile the remaining runtime overhead.
