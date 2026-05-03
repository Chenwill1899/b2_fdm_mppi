# Stage 6 Runtime Profiling / Optimization

Date: 2026-05-03

Branch:

```text
codex/s6-runtime-profiling
```

Base:

```text
origin/fdm @ 16b043b after PR #27 was merged into fdm
```

## Goal

Start S6-002 by profiling the learned Torch/CUDA runtime bottleneck and applying one low-risk optimization without changing MPPI behavior or learned-FDM semantics.

## Root Cause Evidence

Stage 5-D/S6 result packages showed learned Torch runtime around `52-61 ms` per MPPI step versus nominal CUDA around `5.8-6.4 ms`. Existing runtime buckets already pointed to rollout-side terrain feature computation as the dominant cost, followed by FDM inference, state integration, and obstacle cost.

Fresh 20-step profile commands:

```bash
python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle.yaml --output results/stage6_runtime_profile/standard_seed123_g05_20steps --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5
python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_random100_dataset.yaml --output results/stage6_runtime_profile/id_random_seed123_g05_20steps --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5
python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_low_friction_patch.yaml --output results/stage6_runtime_profile/low_friction_seed123_g05_20steps --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5
```

Selected pre-optimization profile buckets:

| Scenario | rollout_total_ms | terrain_features_ms | fdm_inference_ms | state_integrate_ms | obstacle_cost_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 992.23 | 357.61 | 177.20 | 163.97 | 32.47 |
| ID random auto-seed | 1315.42 | 784.27 | 142.68 | 155.78 | 68.05 |
| low_friction_patch | 1072.69 | 445.62 | 162.27 | 166.90 | 6.80 |

The ID random config uses terrain noise and showed the highest terrain-feature bucket. This made the repeated noise bilinear sampling path the first optimization target.

## Optimization

`MppiOmniTorch._terrain_features_torch()` previously sampled the terrain noise grid, x-gradient grid, and y-gradient grid through three independent `_bilinear_sample_torch()` calls. Each call recomputed the same grid coordinates, integer indices, and interpolation weights.

The new `_bilinear_sample_many_torch()` computes those coordinates once and samples the stacked `noise/grad_x/grad_y` tensor in one pass.

Behavior guard:

```bash
python3 -m pytest tests/test_mppi_omni_torch.py::test_nominal_torch_bilinear_sample_many_matches_individual_samples -q
```

Result:

```text
1 passed
```

Targeted regression:

```bash
python3 -m pytest tests/test_mppi_omni_torch.py::test_nominal_torch_batch_cost_matches_numpy_with_terrain_risk tests/test_mppi_omni_learned_torch.py::test_learned_torch_rollout_scales_residual_with_gain -q
```

Result:

```text
2 passed
```

Microbenchmark on CPU with `1024` query points and `32x32` noise grids:

| Method | Mean ms / call |
| --- | ---: |
| three individual samplers | 0.369392 |
| batched sampler | 0.187556 |

Speedup: `1.970x` for the noise-grid sampling subroutine.

## Post-Optimization Profile

Post-optimization profile commands:

```bash
python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle.yaml --output results/stage6_runtime_profile/standard_seed123_g05_20steps_batched_noise --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5
python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_random100_dataset.yaml --output results/stage6_runtime_profile/id_random_seed123_g05_20steps_batched_noise --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5
python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_low_friction_patch.yaml --output results/stage6_runtime_profile/low_friction_seed123_g05_20steps_batched_noise --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5
```

Selected before/after buckets:

| Scenario | Bucket | Before ms | After ms | Delta |
| --- | --- | ---: | ---: | ---: |
| standard | terrain_features_ms | 357.61 | 360.40 | +0.78% |
| ID random auto-seed | terrain_features_ms | 784.27 | 525.28 | -33.02% |
| low_friction_patch | terrain_features_ms | 445.62 | 465.52 | +4.47% |

The measured rollout improvement is visible on the noise-heavy ID random run. The standard and low-friction runs are within profiling variance because they do not exercise the same repeated noise sampling bottleneck as strongly.

Important caveat: `config/b2_omni_oracle_random100_dataset.yaml` uses `scenario.random_seed: auto`; the pre/post ID random profile is useful for hotspot direction but is not a strictly paired trajectory comparison. The microbenchmark and unit equivalence test are the stricter evidence for the sampling change.

## Post-Merge Smoke

After PR #27 was merged into `fdm`, the runtime branch was fast-forwarded to `origin/fdm @ 16b043b`, the S6 optimization stash was reapplied, and a short profiler smoke was run on the merged base:

```bash
python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_random100_dataset.yaml --output results/stage6_runtime_profile/post_merge_id_random_batched_noise_5steps --steps 5 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5
```

Output: `results/stage6_runtime_profile/post_merge_id_random_batched_noise_5steps/`.

Key smoke profile buckets:

| Bucket | Total ms | Mean ms |
| --- | ---: | ---: |
| rollout_total_ms | 273.69 | 54.74 |
| terrain_features_ms | 166.17 | 1.33 |
| fdm_inference_ms | 41.39 | 0.33 |
| obstacle_cost_ms | 35.35 | 7.07 |

## Fixed-Seed 2x2 Runtime Matrix

The original single-controller profiler was useful for hotspot direction but could not directly compare nominal Torch, learned Torch, risk off, and risk on under identical episode seeds. `tools/profile_stage6_runtime_matrix.py` now runs the official 2x2 runtime profiling matrix:

- nominal Torch, risk off;
- nominal Torch, risk on;
- learned Torch balanced, risk off;
- learned Torch balanced, risk on.

The tool forces explicit `scenario.random_seed = base_seed + episode_id` for random-start-goal configs and enables the appropriate Torch controller profile path for both nominal (`mppi.profile_enabled`) and learned (`fdm.profile_enabled`) controllers.

Test:

```bash
python3 -m pytest tests/test_stage6_runtime_matrix.py tests/test_stage5_runtime_profile.py -q
```

Result:

```text
2 passed
```

Real paired profiler smoke:

```bash
python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed --output results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps --episodes 3 --steps 5 --base-seed 123 --backend torch --device auto --risk-weight 3 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
```

Output: `results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps/stage6_runtime_matrix_summary.json`.

Aggregate profile means:

| Case | mean_mppi_time_ms | profile_mean_rollout_total_ms | terrain_features_ms | fdm_inference_ms | terrain_risk_cost_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal_risk_off | 23.76 | 10.71 | n/a | n/a | 0.053 |
| nominal_risk_on | 18.87 | 10.28 | n/a | n/a | 4.424 |
| learned_risk_off | 44.55 | 40.35 | 0.928 | 0.232 | 0.054 |
| learned_risk_on | 44.09 | 38.85 | 0.913 | 0.192 | 1.122 |

Paired deltas from the same run:

| Pair | mean_mppi_time_ms_delta | rollout_total_ms_delta | terrain_features_ms_delta | fdm_inference_ms_delta | terrain_risk_cost_ms_delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal_risk_on - nominal_risk_off | -4.89 | -0.43 | n/a | n/a | +4.371 |
| learned_risk_on - learned_risk_off | -0.46 | -1.50 | -0.015 | -0.040 | +1.068 |
| learned_risk_off - nominal_risk_off | +20.78 | +29.64 | n/a | n/a | +0.001 |
| learned_risk_on - nominal_risk_on | +25.22 | +28.57 | n/a | n/a | -3.302 |

Interpretation boundary: this is a short profiling run, not a paper runtime benchmark. It confirms the next large bottleneck is still learned rollout work: learned risk-on adds about `+25.22 ms` mean MPPI time and `+28.57 ms` profiled rollout time over nominal risk-on under matched seeds. Terrain-risk cost itself is not the main learned overhead; FDM rollout/feature/integration work remains the target.

Profiler delta enhancement:

```bash
python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed_delta_buckets --output results/stage6_runtime_profile/paired_id_random_fixed_seed_1ep_2steps_delta_buckets --episodes 1 --steps 2 --base-seed 123 --backend torch --device cuda --risk-weight 3.0
```

Output: `results/stage6_runtime_profile/paired_id_random_fixed_seed_1ep_2steps_delta_buckets/stage6_runtime_matrix_summary.json`.

The paired delta JSON now includes:

- `profile_mean_sample_candidates_ms_delta_mean`
- `profile_mean_update_distribution_ms_delta_mean`

This makes first-call sampling/update overhead visible in paired comparisons, which matters when interpreting very short profiling runs.

## Torch Inference Mode Guard

The next low-risk runtime cleanup disables autograd for the entire Torch MPPI `compute_control()` path. Rollout already used a local `torch.no_grad()` block, but candidate sampling, cost aggregation, distribution update, and the learned controller entry path still ran under the caller's default grad-enabled context.

TDD guard:

```bash
python3 -m pytest tests/test_mppi_omni_torch.py::test_nominal_torch_compute_control_disables_grad_tracking -q
```

Red result before implementation:

```text
assert [True] == [False]
```

Green result after wrapping `MppiOmniTorch.compute_control()` in `torch.inference_mode()`:

```text
1 passed
```

Targeted regression:

```bash
python3 -m pytest tests/test_mppi_omni_torch.py tests/test_mppi_omni_learned_torch.py tests/test_stage6_runtime_matrix.py -q
```

Result:

```text
17 passed
```

Fixed-seed profiler command:

```bash
python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed --output results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode --episodes 3 --steps 5 --base-seed 123 --backend torch --device cuda --risk-weight 3.0
```

Output: `results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode/stage6_runtime_matrix_summary.json`.

Matched short-run before/after aggregates:

| Case | mean_mppi before | mean_mppi after | rollout before | rollout after | terrain_features before | terrain_features after | fdm_inference before | fdm_inference after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal_risk_off | 23.76 | 22.75 | 10.71 | 9.43 | n/a | n/a | n/a | n/a |
| nominal_risk_on | 18.87 | 17.17 | 10.28 | 8.73 | n/a | n/a | n/a | n/a |
| learned_risk_off | 44.55 | 40.80 | 40.35 | 36.66 | 0.928 | 0.850 | 0.232 | 0.223 |
| learned_risk_on | 44.09 | 40.86 | 38.85 | 35.74 | 0.913 | 0.852 | 0.192 | 0.183 |

Paired learned risk-on minus nominal risk-on changed from `+25.22 ms` to `+23.69 ms` in mean MPPI time and from `+28.57 ms` to `+27.01 ms` in profiled rollout time.

Interpretation boundary: this is a behavior-preserving runtime hygiene change with modest measured benefit on the short profiler. It does not close the learned-vs-nominal runtime gap. The next meaningful optimization still needs to attack the learned rollout loop, especially per-horizon terrain feature and residual model calls.

## Boundary

This is a first S6-002 optimization, not a complete runtime closure. Learned Torch is still slower than nominal. Remaining likely hotspots:

- per-horizon FDM inference and terrain feature calls inside the rollout loop;
- state integration loop over horizon;
- random-obstacle obstacle cost in dense maps;
- profiling synchronization overhead when `profile_enabled=true`.

Next optimization candidates:

- expand the paired runtime profiler to longer fixed-seed runs after the next optimization candidate;
- investigate Torch compilation or horizon-loop fusion for learned rollout;
- cache or reuse terrain/risk features only where timestep semantics match;
- profile obstacle cost separately on dense random maps.

## Stage 6 Closeout Figures

Stage 6 runtime closeout adds a dedicated plotting script:

```bash
python3 tools/plot_stage6_runtime_results.py --summary results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode/stage6_runtime_matrix_summary.json results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout/stage6_runtime_matrix_summary.json --output figures/stage6 --tables-output tables/stage6
```

Generated tracked figures:

- `figures/stage6/fig_stage6_runtime_summary.png`
- `figures/stage6/fig_stage6_runtime_summary.pdf`
- `figures/stage6/fig_stage6_runtime_breakdown.png`
- `figures/stage6/fig_stage6_runtime_breakdown.pdf`
- `figures/stage6/fig_stage6_runtime_delta_buckets.png`
- `figures/stage6/fig_stage6_runtime_delta_buckets.pdf`

Generated tracked tables:

- `tables/stage6/table_stage6_runtime_summary.csv`
- `tables/stage6/table_stage6_runtime_paired_deltas.csv`

Closeout profiler command:

```bash
python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed_closeout --output results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout --episodes 10 --steps 10 --base-seed 123 --backend torch --device cuda --risk-weight 3.0
```

Output: `results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout/stage6_runtime_matrix_summary.json`.

Closeout aggregate means:

| Case | mean_mppi_time_ms | rollout_total_ms | terrain_features_ms | fdm_inference_ms | terrain_risk_cost_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal_risk_off | 14.27 | 8.74 | n/a | n/a | 0.052 |
| nominal_risk_on | 14.31 | 8.58 | n/a | n/a | 1.640 |
| learned_risk_off | 40.57 | 36.39 | 0.867 | 0.195 | 0.050 |
| learned_risk_on | 41.38 | 36.17 | 0.865 | 0.187 | 1.085 |

Closeout paired deltas:

| Pair | mean_mppi_time_ms_delta | rollout_total_ms_delta | sample_candidates_delta | update_distribution_delta | terrain_risk_cost_delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_risk_on - nominal_risk_on | +27.08 | +27.58 | +0.001 | -0.006 | -0.555 |
| learned_risk_off - nominal_risk_off | +26.30 | +27.65 | -0.388 | -0.422 | -0.002 |
| learned_risk_on - learned_risk_off | +0.81 | -0.22 | +0.005 | +0.013 | +1.035 |

Interpretation: Stage 6 now has a reproducible runtime profiling package and paper-style runtime figures. It does not make a real-time learned-FDM-MPPI claim. The result is a bounded limitation: same-backend learned Torch remains about `+27 ms` slower than nominal Torch on the closeout profile, and the dominant gap is learned rollout rather than terrain-risk cost, sample generation, or distribution update.
