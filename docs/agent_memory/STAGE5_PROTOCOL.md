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
  residual_gain: 1.0
  profile_enabled: false
```

`residual_gain` scales the learned residual before rollout integration:

```text
real_control = clip(response_command + residual_gain * du_hat)
```

`residual_gain=0.0` keeps the learned backend, artifact loading, terrain features, and Torch rollout path active but disables residual correction. It is the Stage 5-C backend control group.

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
  --fdm-device cpu \
  --fdm-residual-gain 1.0
```

If `fdm.enabled=true` with `mppi.backend=cuda`, the learned controller uses the Torch rollout backend from PR #19. If CUDA is requested but unavailable, controller creation fails with a clear Torch CUDA availability error.

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
  --fdm-device cpu \
  --fdm-residual-gain 1.0
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

## Stage 5-C Calibration And Profiling

Stage 5-C starts after PR #18 and PR #19 are merged. It keeps the MLP residual FDM fixed and calibrates closed-loop use of that model before adding history-conditioned FDM.

Residual-gain sweep tool:

```text
tools/sweep_stage5_calibration.py
```

Runtime profile tool:

```text
tools/profile_stage5_learned_torch.py
```

Primary question:

```text
Does random-task degradation come from full residual correction being too strong,
or from the MLP residual model itself?
```

The first ablation must include:

```text
residual_gain = 0.0 / 0.25 / 0.5 / 0.75 / 1.0
```

Cost calibration should be small and learned-controller-only at first. Supported override keys:

```text
goal_xy_weight
obstacle_weight
obstacle_soft_weight
smooth_weight
accel_weight
lateral_weight
yaw_rate_weight
```

## Visual Evaluation Protocol

Use the visual evaluation tool when inspecting the closed-loop learning effect for a single scenario:

```text
tools/visualize_stage5_closed_loop.py
```

The detailed visual protocol lives in:

```text
docs/agent_memory/STAGE5_VISUAL_EVAL.md
```

It runs paired `Nominal-MPPI` and `Learned-FDM-MPPI` oracle-world simulations with the same config and seed, enables `trajectory.png` and `animation.gif` for both runs, and writes:

```text
closed_loop_nominal_vs_learned.png
closed_loop_compare_metrics.csv
closed_loop_compare_metrics.json
stage5_visual_eval_summary.json
```

This visual check is useful for human inspection, but benchmark conclusions still come from `tools/benchmark_learned_fdm_mppi.py`.

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

PR #18 is only the benchmark tool and schema. PR #19 adds a Torch CUDA learned rollout backend, runs the first standard/ID/OOD closed-loop benchmark, and summarizes results. PR #20 should profile and tune learned-FDM runtime and closed-loop cost calibration. Do not start history-conditioned FDM until closed-loop benchmark evidence shows the current MLP residual model is insufficient.

PR #19 result boundary:

- standard scene: learned improves final distance, steps, clearance, terrain risk, and smoothness.
- ID/OOD random tasks: learned reaches 100% success but does not stably outperform nominal on final distance, steps, or clearance.
- learned consistently reduces terrain risk, command-real error, residual norm, smoothness, and jerk.
- Torch CUDA learned rollout makes benchmark practical, but is still slower than nominal CUDA.

S5-008 calibrated result boundary:

- Calibrated learned `residual_gain=0.5`, `goal_xy_weight=3.5`, `smooth_weight=1.0` reaches 100% success on ID, OOD obstacle, and OOD terrain 20-episode suites.
- It removes the default learned `residual_gain=1.0` regression on random-task final distance and steps:
  - ID random: final distance `0.6766` vs nominal `0.6795`, steps `129.8` vs nominal `137.1`.
  - OOD obstacle: final distance `0.6775` vs nominal `0.6768`, steps `130.4` vs nominal `142.3`.
  - OOD terrain: final distance `0.6785` vs nominal `0.6848`, steps `130.2` vs nominal `141.8`.
- The calibration trades away part of default learned `residual_gain=1.0`'s terrain-risk and smoothness advantage. Treat default learned as the conservative/smooth reference and calibrated learned as the efficiency candidate.
- Stage 5-D should include at least nominal CUDA, learned default `residual_gain=1.0`, and calibrated learned `residual_gain=0.5`, `goal_xy_weight=3.5`, `smooth_weight=1.0`. If the paper claim needs one learned controller to improve both efficiency and risk/smoothness, run a small Pareto cost sweep before expanding to 50-100 episodes.

S5-009 Pareto result boundary:

- The full ID grid used `residual_gain=0.4/0.5/0.6`, `goal_xy_weight=3.0/3.5/4.0`, and `smooth_weight=0.75/1.0/1.25` for `10` episodes per case.
- OOD validation then used three selected candidates for `10` episodes each on OOD obstacle and OOD terrain.
- `residual_gain=0.5`, `goal_xy_weight=3.0`, `smooth_weight=0.75` is the current balanced candidate. Across ID/OOD obstacle/OOD terrain, mean deltas versus nominal are final distance `-0.0025`, steps `-5.57`, terrain risk `+0.0023`, smoothness `+0.000081`, and jerk `+0.000046`.
- `residual_gain=0.6`, `goal_xy_weight=4.0`, `smooth_weight=1.0` is the aggressive efficiency candidate. It improves mean final distance by `-0.0066` and steps by `-13.20`, but has larger risk/smoothness/jerk penalties.
- Stage 5-D should compare nominal CUDA, default learned `residual_gain=1.0`, current efficiency `0.5/3.5/1.0`, balanced `0.5/3.0/0.75`, and optionally aggressive efficiency `0.6/4.0/1.0`.
- History-conditioned FDM is still deferred. The current MLP-FDM has not yet failed calibrated closed-loop evaluation strongly enough to justify changing model structure.

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
