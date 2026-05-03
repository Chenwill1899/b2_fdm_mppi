# Stage 5 Closed-loop Visual Evaluation Protocol

Last updated: 2026-05-01

## Goal

This protocol standardizes how to visually inspect the closed-loop effect of learned FDM inside MPPI.

The key distinction is:

- **Open-loop replay** checks whether learned FDM predicts oracle execution for a fixed command sequence.
- **Closed-loop visual eval** checks what the controller actually executes after MPPI replans at every step.

Use closed-loop visual eval when asking "learning 前后效果怎么看". Use open-loop replay only as supporting model-prediction evidence.

## Tool

Tool:

```text
tools/visualize_stage5_closed_loop.py
```

Default standard-scene command:

```bash
python3 tools/visualize_stage5_closed_loop.py \
  --config config/b2_omni_oracle.yaml \
  --scenario-name standard \
  --output results/stage5_visual_eval/standard_seed123_cuda \
  --seed 123 \
  --backend cuda \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-checkpoint best_model.pt \
  --fdm-normalization normalization.npz \
  --fdm-device cuda
```

The tool runs two oracle-world closed-loop simulations with the same config and seed:

```text
Nominal-MPPI
Learned-FDM-MPPI
```

It enables plots and animation for both runs, then writes a single overlay figure and metric table.

## Outputs

Main output directory:

```text
results/stage5_visual_eval/<scenario>_seed<seed>_<backend>/
```

Expected top-level files:

```text
stage5_visual_eval_summary.json
closed_loop_nominal_vs_learned.png
closed_loop_compare_metrics.csv
closed_loop_compare_metrics.json
```

Expected per-run files:

```text
runs/<scenario>_episode_0000_nominal/summary.json
runs/<scenario>_episode_0000_nominal/trajectory.csv
runs/<scenario>_episode_0000_nominal/trajectory.png
runs/<scenario>_episode_0000_nominal/animation.gif

runs/<scenario>_episode_0000_learned/summary.json
runs/<scenario>_episode_0000_learned/trajectory.csv
runs/<scenario>_episode_0000_learned/trajectory.png
runs/<scenario>_episode_0000_learned/animation.gif
```

## How To Read The Result

Primary visual:

```text
closed_loop_nominal_vs_learned.png
```

Color convention:

```text
blue   = Nominal-MPPI closed-loop execution
orange = Learned-FDM-MPPI closed-loop execution
gray   = obstacle body
red dashed circle = obstacle radius + robot radius + safety_dist
```

Primary metrics:

```text
reached_goal
failed
steps
run_time
final_distance
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

The metric table reports `learned_minus_nominal`. Lower is better for:

```text
failed
steps
run_time
final_distance
path_length
mean_terrain_risk
mean_cmd_real_error
mean_residual_norm
control_smoothness
control_jerk
mean_mppi_time_ms
max_mppi_time_ms
```

Higher is better for:

```text
reached_goal
min_obstacle_clearance
```

## Current Standard Scene Visual Check

Latest manual visual check before this protocol was standardized:

```text
Nominal output: results/sim_results/b2_omni_oracle_2026-05-01_22-48-39
Learned output: results/sim_results/b2_omni_oracle_2026-05-01_22-44-09
Overlay: results/stage5_visual_compare_seed123/closed_loop_nominal_vs_learned.png
```

Metrics:

| Metric | Nominal | Learned | Learned - Nominal |
| --- | ---: | ---: | ---: |
| reached_goal | true | true | n/a |
| failed | false | false | n/a |
| steps | 225 | 214 | -11 |
| run_time | 22.5 | 21.4 | -1.1 |
| final_distance | 0.3461 | 0.3371 | -0.0090 |
| path_length | 18.6293 | 18.2588 | -0.3706 |
| min_obstacle_clearance | 0.1198 | 0.2024 | +0.0826 |
| mean_terrain_risk | 0.4289 | 0.4140 | -0.0149 |
| control_smoothness | 0.003071 | 0.002760 | -0.000311 |
| control_jerk | 0.003512 | 0.002405 | -0.001107 |
| mean_mppi_time_ms | 7.18 | 27.73 | +20.55 |

Interpretation:

- Standard scene: learned-FDM closed-loop is visually cleaner around obstacles and reaches the goal with fewer steps.
- It improves final distance, path length, clearance, terrain risk, smoothness, and jerk.
- It is still slower than nominal CUDA.
- This is a single-scene visual check, not enough for a full Stage 5 improvement claim. Use `docs/agent_memory/STAGE5_BENCHMARK.md` for ID/OOD benchmark conclusions.

## Supporting Open-loop Replay

For model-prediction visualization, run:

```bash
python3 tools/evaluate_residual_fdm_rollout.py \
  --config config/b2_omni_oracle.yaml \
  --model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --output results/fdm_rollout_eval/stage5_visual_check_seed123 \
  --seed 123 \
  --backend numpy \
  --device cuda \
  --checkpoint best_model.pt \
  --normalization normalization.npz \
  --gif-fps 8 \
  --gif-max-frames 120
```

Open-loop outputs:

```text
trajectory_compare.png
residual_compare.png
rollout_compare.gif
rollout_metrics.json
```

Current open-loop standard check:

```text
nominal ADE/FDE: 0.5213 / 0.9190 m
learned ADE/FDE: 0.0484 / 0.0823 m
ADE improvement: 90.71%
FDE improvement: 91.04%
residual MSE improvement vs zero residual: 98.53%
```

## Boundary

Closed-loop visual eval is a qualitative and single-seed inspection layer. It should be used with the Stage 5 benchmark summary, not instead of it.

Current Stage 5 status:

- Learned-FDM-MPPI has a working CUDA closed-loop reference path.
- Standard scene visual check is positive.
- ID/OOD random-task benchmark does not yet show stable final-distance or step-count improvement over nominal.
- Next work should focus on profiling, rollout/cost calibration, and runtime optimization before stronger Stage 5 claims.
