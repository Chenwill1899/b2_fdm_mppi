# Stage 4 Residual FDM Protocol

Last updated: 2026-05-01

## Goal

Stage 4 validates whether a one-step residual velocity FDM is learnable, reproducible, and useful for open-loop trajectory replay.

Stage 4 does not validate closed-loop Learned-FDM-MPPI. Stage 5 can start only after the residual FDM baseline is stable on ID data, acceptable on OOD/cross-map data, and clearly improves open-loop ADE/FDE over nominal replay.

## Dataset Protocol

ID dataset generation:

```bash
python3 tools/generate_oracle_episodes.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --episodes 500 \
  --base-seed 123 \
  --output datasets/oracle_stage4 \
  --backend numpy \
  --num-workers 8
```

Build ID train/val/test split:

```bash
python3 tools/build_oracle_dataset.py \
  --input datasets/oracle_stage4 \
  --output datasets/oracle_stage4_splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 123
```

Validate ID split:

```bash
python3 tools/validate_oracle_dataset.py \
  --dataset datasets/oracle_stage4_splits \
  --output datasets/oracle_stage4_splits
```

The split is episode-level. Transitions from the same episode do not cross train/val/test, but all ID splits share the same fixed obstacle map and terrain seed.

## Training Protocol

Model: one-step MLP residual regressor.

Input:

```text
features = concat(states, cmd_controls, terrain_features, terrain_risk)
shape    = [B, 14]
```

Target:

```text
exec_residuals = real_controls - cmd_controls
shape          = [B, 3]
```

Hardened seed training command:

```bash
python3 tools/train_residual_fdm.py \
  --dataset datasets/oracle_stage4_splits \
  --output results/fdm_baselines/stage4_mlp_seed123_hardened \
  --epochs 50 \
  --batch-size 512 \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --seed 123 \
  --device cpu
```

Multi-seed benchmark uses seeds `123`, `456`, and `789` with identical hyperparameters and output directories:

```text
results/fdm_baselines/stage4_mlp_seed123_hardened
results/fdm_baselines/stage4_mlp_seed456_hardened
results/fdm_baselines/stage4_mlp_seed789_hardened
```

## Checkpoint Policy

```text
best_model.pt = checkpoint at minimum validation standardized loss
model.pt      = final epoch checkpoint
```

`metrics.json` must record `best_epoch`, `best_val_loss`, `final_epoch`, `final_val_loss`, `checkpoint_policy`, `best_checkpoint_path`, and `final_checkpoint_path`.

## Metrics

Training metrics:

- val/test raw MSE.
- per-axis MSE and RMSE for `vx`, `vy`, `wz`.
- zero-residual baseline MSE.
- per-axis MSE reduction percentage.
- overall MSE reduction percentage.
- overall improvement multiplier over zero-residual baseline.
- exact command, argv, git SHA/branch/dirty flag, device, dataset artifact paths, and output paths.

Open-loop rollout metrics:

- nominal vs learned ADE/FDE.
- ADE/FDE at `1s`, `2s`, and `4s`.
- residual MSE vs zero-residual MSE.
- exact command, git SHA/branch/dirty flag, device, checkpoint path, normalization path, and GIF parameters.

## Open-loop Rollout Eval Protocol

Standard scene:

```text
config/b2_omni_oracle.yaml
```

Best-checkpoint command:

```bash
python3 tools/evaluate_residual_fdm_rollout.py \
  --config config/b2_omni_oracle.yaml \
  --model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --output results/fdm_rollout_eval/stage4_mlp_seed123_hardened_b2_omni_oracle_seed123 \
  --seed 123 \
  --backend numpy \
  --device cpu \
  --checkpoint best_model.pt \
  --normalization normalization.npz \
  --gif-fps 8 \
  --gif-max-frames 120
```

## OOD Protocol

OOD obstacle config:

```text
config/b2_omni_oracle_random100_dataset_ood_obstacle.yaml
```

Changes obstacle seed and obstacle distribution while keeping terrain seed fixed.

OOD terrain config:

```text
config/b2_omni_oracle_random100_dataset_ood_terrain.yaml
```

Changes terrain seed and slightly changes terrain scale while keeping obstacle seed fixed.

OOD dataset commands:

```bash
python3 tools/generate_oracle_episodes.py \
  --config config/b2_omni_oracle_random100_dataset_ood_obstacle.yaml \
  --episodes 100 \
  --base-seed 123 \
  --output datasets/oracle_stage4_ood_obstacle \
  --backend numpy \
  --num-workers 8

python3 tools/generate_oracle_episodes.py \
  --config config/b2_omni_oracle_random100_dataset_ood_terrain.yaml \
  --episodes 100 \
  --base-seed 123 \
  --output datasets/oracle_stage4_ood_terrain \
  --backend numpy \
  --num-workers 8
```

Build and validate:

```bash
python3 tools/build_oracle_dataset.py \
  --input datasets/oracle_stage4_ood_obstacle \
  --output datasets/oracle_stage4_ood_obstacle_splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 123

python3 tools/validate_oracle_dataset.py \
  --dataset datasets/oracle_stage4_ood_obstacle_splits \
  --output datasets/oracle_stage4_ood_obstacle_splits

python3 tools/build_oracle_dataset.py \
  --input datasets/oracle_stage4_ood_terrain \
  --output datasets/oracle_stage4_ood_terrain_splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 123

python3 tools/validate_oracle_dataset.py \
  --dataset datasets/oracle_stage4_ood_terrain_splits \
  --output datasets/oracle_stage4_ood_terrain_splits
```

OOD residual eval:

```bash
python3 tools/evaluate_residual_fdm_dataset.py \
  --dataset datasets/oracle_stage4_ood_obstacle_splits \
  --model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --output results/fdm_ood_eval/ood_obstacle_seed123 \
  --checkpoint best_model.pt \
  --normalization normalization.npz \
  --device cpu
```

OOD rollout eval uses the OOD config as `--config` and the same `best_model.pt`. OOD tests measure generalization only; do not retrain on OOD before reporting the baseline.

## Current Results

Seed123 hardened ID baseline:

```text
val_mse: 1.000058e-05
test_mse: 1.005117e-05
zero_residual_val_mse: 6.534630e-04
zero_residual_test_mse: 6.288914e-04
overall_val_improvement_x: 65.34
overall_test_improvement_x: 62.57
overall_test_mse_reduction_pct: 98.40
best_epoch: 46
final_epoch: 50
```

Multi-seed benchmark, seeds `123/456/789`:

```text
mean_test_mse: 1.014878e-05
std_test_mse: 2.183396e-07
mean_overall_test_mse_reduction_pct: 98.3862
std_overall_test_mse_reduction_pct: 0.0347
mean_overall_test_improvement_x: 61.9955
```

Standard open-loop best-checkpoint eval:

```text
nominal_ade_xy: 0.521254
learned_ade_xy: 0.048437
nominal_fde_xy: 0.919049
learned_fde_xy: 0.082303
residual_mse_improvement_pct: 98.5339
```

OOD obstacle:

```text
residual test_mse: 9.931577e-06
residual test improvement: 57.20x
rollout learned ADE/FDE: 0.020705 / 0.046311
rollout nominal ADE/FDE: 0.228230 / 0.328461
```

OOD terrain:

```text
residual test_mse: 1.073199e-05
residual test improvement: 59.64x
rollout learned ADE/FDE: 0.024651 / 0.053852
rollout nominal ADE/FDE: 0.264790 / 0.394505
```

## Paper Table Template

| Setting | Seed(s) | Val MSE | Test MSE | Zero Test MSE | Test Reduction % | Test Improvement x | Learned ADE | Nominal ADE | Learned FDE | Nominal FDE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID | 123 | 1.000e-05 | 1.005e-05 | 6.289e-04 | 98.40 | 62.57 | 0.0484 | 0.5213 | 0.0823 | 0.9190 |
| ID mean | 123/456/789 | 1.005e-05 | 1.015e-05 | 6.289e-04 | 98.39 | 62.00 | - | - | - | - |
| OOD obstacle | 123 | 9.229e-06 | 9.932e-06 | 5.681e-04 | 98.25 | 57.20 | 0.0207 | 0.2282 | 0.0463 | 0.3285 |
| OOD terrain | 123 | 9.493e-06 | 1.073e-05 | 6.401e-04 | 98.32 | 59.64 | 0.0247 | 0.2648 | 0.0539 | 0.3945 |

## Stage 5 Entry Condition

- ID multi-seed baseline remains stable.
- OOD obstacle and OOD terrain results remain meaningfully better than zero-residual and nominal replay.
- Standard open-loop ADE/FDE clearly improves over nominal.
- `best_model.pt` is the default checkpoint for Stage 5 experiments.
- Only after learned FDM is integrated into MPPI rollout and closed-loop eval is complete should the project claim Stage 5 validation.
