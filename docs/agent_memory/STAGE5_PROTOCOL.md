# Stage 5 Learned-FDM-MPPI Protocol

Last updated: 2026-05-01

## Goal

Stage 5 integrates the trained residual FDM into MPPI rollout dynamics so sampled trajectories are scored under learned execution response, not only nominal dynamics.

Stage 5 entry starts with NumPy-only smoke tests. CUDA rollout integration, large ID/OOD closed-loop benchmarks, and final Learned-FDM-MPPI claims are later work.

## Learned Residual Dynamics Interface

Wrapper:

```text
b2_fdm_mppi/core/learned_residual_dynamics.py
```

Artifacts:

```text
best_model.pt
normalization.npz
feature_mean / feature_std
target_mean / target_std
```

Core API:

```python
LearnedResidualDynamics.from_artifacts(
    model_dir,
    robot,
    terrain,
    checkpoint="best_model.pt",
    normalization="normalization.npz",
    device="cpu",
)

predict_residual(state, command, terrain_features=None, terrain_risk=None)
predict_residual_batch(states, commands, terrain_features=None, terrain_risk=None)
step(state, command)
```

`command` is the controller-side response-limited velocity command. The wrapper predicts `du_hat`, applies `real_control = clip(command + du_hat)`, and integrates with `OmniB2.update_state()`.

## MPPI Rollout Integration

First integration path:

```text
b2_fdm_mppi/controllers/mppi_omni_learned_numpy.py
```

`LearnedFdmMppiOmniNumpy` subclasses `MppiOmniNumpy` and only overrides `_rollout_batch()`. Sampling, cost weighting, nominal control shifting, obstacle costs, and runner execution remain unchanged.

Rollout step:

```text
sampled control
  -> velocity response / acceleration limit
  -> learned residual prediction
  -> real_control = clip(response_command + du_hat)
  -> SE(2) integration
```

CUDA learned rollout is intentionally unsupported in PR #17.

## Config And CLI

YAML block:

```yaml
fdm:
  enabled: true
  model_dir: results/fdm_baselines/stage4_mlp_seed123_hardened
  checkpoint: best_model.pt
  normalization: normalization.npz
  device: cpu
```

CLI smoke command:

```bash
python3 tools/run_omni_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --seed 123 \
  --backend numpy \
  --fdm-enabled \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-checkpoint best_model.pt \
  --fdm-normalization normalization.npz \
  --fdm-device cpu
```

If `fdm.enabled=true` with `mppi.backend=cuda`, controller creation must fail with a clear NumPy-only error.

## Closed-loop Smoke Protocol

Run paired standard-scene oracle simulations:

```bash
python3 tools/run_omni_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --seed 123 \
  --backend numpy

python3 tools/run_omni_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --seed 123 \
  --backend numpy \
  --fdm-enabled \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-checkpoint best_model.pt \
  --fdm-normalization normalization.npz \
  --fdm-device cpu
```

Required outputs:

```text
summary.json
trajectory.csv
controls.csv
residuals.csv
terrain.csv
time_results.csv
```

Smoke acceptance:

- learned run has no NaN/Inf and `failed=false`.
- learned run records finite `mean_mppi_time_ms` and `max_mppi_time_ms`.
- learned `final_distance <= nominal_final_distance * 1.2 + 0.1m`.
- result remains a Stage 5 entry smoke, not a full Stage 5 benchmark.

## Current PR #17 Smoke Result

Nominal NumPy oracle run:

```text
result: results/sim_results/b2_omni_oracle_2026-05-01_20-41-06
reached_goal: true
failed: false
steps: 219
final_distance: 0.3281706
min_obstacle_clearance: 0.1524212
mean_mppi_time_ms: 13.4969
max_mppi_time_ms: 25.4927
```

Learned-FDM NumPy oracle run:

```text
result: results/sim_results/b2_omni_oracle_2026-05-01_20-42-31
reached_goal: true
failed: false
steps: 209
final_distance: 0.3429893
min_obstacle_clearance: 0.1900530
mean_mppi_time_ms: 1143.4956
max_mppi_time_ms: 1224.5321
```

Acceptance threshold:

```text
nominal_final_distance * 1.2 + 0.1 = 0.4938048
learned_final_distance = 0.3429893
```

The learned controller passes the stability smoke but is not real-time in this first NumPy implementation. The high runtime is expected from Python terrain feature evaluation plus per-horizon Torch inference in the rollout loop.

## Benchmark Protocol

PR #18 adds the first benchmark runner:

```text
tools/benchmark_learned_fdm_mppi.py
```

The detailed protocol and output schema live in:

```text
docs/agent_memory/STAGE5_BENCHMARK.md
```

Run closed-loop ID/OOD benchmarks only after the NumPy smoke is stable:

- ID: `config/b2_omni_oracle_random100_dataset.yaml`
- OOD obstacle: `config/b2_omni_oracle_random100_dataset_ood_obstacle.yaml`
- OOD terrain: `config/b2_omni_oracle_random100_dataset_ood_terrain.yaml`

Compare at least:

```text
Nominal-MPPI in oracle world
Learned-FDM-MPPI in oracle world
Oracle-MPPI upper bound, if implementation cost is reasonable
```

Core metrics:

```text
success_rate
final_distance
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

The four gate metrics are `success_rate`, `final_distance`, `min_obstacle_clearance`, and `mean_mppi_time_ms`.

PR #18 is only the benchmark tool and schema. PR #19 should run the full ID/OOD benchmark and summarize results. PR #20 should profile and optimize learned-FDM runtime. Do not start history-conditioned FDM until closed-loop benchmark evidence shows the current MLP residual model is insufficient.

## Failure Modes

- Missing checkpoint or normalization artifact.
- Feature-name or input-dimension mismatch between checkpoint and Stage 4 schema.
- NaN/Inf from model prediction or rollout integration.
- Learned rollout improves final distance but violates clearance.
- Learned rollout is stable but too slow for closed-loop use.
- CUDA config accidentally enables FDM before CUDA learned rollout exists.

## Stage 6 Entry Condition

Stage 6 can start only after:

- NumPy learned closed-loop smoke is stable.
- ID benchmark shows learned controller is not worse than nominal on success/final distance/clearance.
- OOD benchmark does not show unacceptable regression.
- Runtime limitation is either optimized or explicitly scoped as offline evaluation.
- No Stage 5 claim is made from open-loop replay alone.
