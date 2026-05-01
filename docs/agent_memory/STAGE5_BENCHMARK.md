# Stage 5-B Closed-loop Benchmark Protocol

Last updated: 2026-05-01

## Goal

Stage 5-B tests whether the Stage 4 MLP residual FDM improves closed-loop MPPI behavior in the oracle world. This is the first benchmark layer after the PR #17 NumPy smoke path.

This protocol is still Stage 5 validation work. It does not claim CUDA learned rollout support, real-time learned-FDM-MPPI, or completion of the full Stage 5 benchmark.

## Benchmark Tool

Tool:

```text
tools/benchmark_learned_fdm_mppi.py
```

Default standard-scene command:

```bash
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --scenario-name standard \
  --output results/stage5_benchmark/standard_seed123 \
  --episodes 1 \
  --base-seed 123 \
  --backend numpy \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-checkpoint best_model.pt \
  --fdm-normalization normalization.npz \
  --fdm-device cpu
```

The tool runs paired `Nominal-MPPI` and `Learned-FDM-MPPI` simulations in `simulation.world_mode: oracle`. It forces `mppi.backend: numpy`, disables plots and animation by default, and writes each run under an episode-scoped directory.

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

PR #18 only adds the benchmark runner and schema. PR #19 should run and summarize the full benchmark:

```bash
# Standard scene
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --scenario-name standard \
  --output results/stage5_benchmark/standard_seed123 \
  --episodes 1 \
  --base-seed 123 \
  --backend numpy \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened

# ID random-task benchmark
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --scenario-name id_random_tasks \
  --output results/stage5_benchmark/id_random_tasks_seed123 \
  --episodes 20 \
  --base-seed 123 \
  --backend numpy \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened

# OOD obstacle benchmark
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle_random100_dataset_ood_obstacle.yaml \
  --scenario-name ood_obstacle \
  --output results/stage5_benchmark/ood_obstacle_seed123 \
  --episodes 20 \
  --base-seed 123 \
  --backend numpy \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened

# OOD terrain benchmark
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle_random100_dataset_ood_terrain.yaml \
  --scenario-name ood_terrain \
  --output results/stage5_benchmark/ood_terrain_seed123 \
  --episodes 20 \
  --base-seed 123 \
  --backend numpy \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened
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

## Runtime Boundary

The PR #17 smoke measured learned-FDM NumPy MPPI at roughly `1143 ms` mean per step versus nominal NumPy at roughly `13.5 ms`. Stage 5-B uses this path as a correctness/reference backend only.

Stage 5-C should profile before optimizing:

```text
terrain feature/risk
FDM Torch inference
Python horizon loop
cost evaluation
array copy / device transfer
```

Do not modify CUDA rollout or replace the MLP model before Stage 5-B evidence is available.
