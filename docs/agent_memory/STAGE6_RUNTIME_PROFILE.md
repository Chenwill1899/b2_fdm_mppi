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

## Boundary

This is a first S6-002 optimization, not a complete runtime closure. Learned Torch is still slower than nominal. Remaining likely hotspots:

- per-horizon FDM inference and terrain feature calls inside the rollout loop;
- state integration loop over horizon;
- random-obstacle obstacle cost in dense maps;
- profiling synchronization overhead when `profile_enabled=true`.

Next optimization candidates:

- add a paired runtime profiler that fixes scenario seeds and compares nominal Torch, learned Torch, risk off, and risk on;
- investigate Torch compilation or horizon-loop fusion for learned rollout;
- cache or reuse terrain/risk features only where timestep semantics match;
- profile obstacle cost separately on dense random maps.
