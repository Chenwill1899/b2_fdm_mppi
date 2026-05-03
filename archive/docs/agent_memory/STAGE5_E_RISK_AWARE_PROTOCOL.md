# Stage 5-E / S6 Risk-Aware Learned-FDM-MPPI Protocol

## Motivation

Stage 5-D showed learned-FDM-MPPI operating modes, but terrain risk was only an FDM input and evaluation metric. Stage 5-E makes terrain risk an explicit MPPI objective, so risk-aware claims are tied to planner cost, not only post-hoc metrics.

## Cost

Official runs use:

```text
terrain_risk_mode=excess
terrain_risk_threshold=0.3
terrain_risk_power=2.0
terrain_risk_weight in {0, selected_weight}
```

For excess mode:

```text
cost += terrain_risk_weight * sum(max(risk(x_t) - threshold, 0) ** power)
```

## Same-Backend 2x2

Official evidence uses `backend=torch` for all four cells:

- nominal Torch, risk off
- nominal Torch, risk on
- learned Torch, risk off
- learned Torch, risk on

Learned balanced setting:

```text
residual_gain=0.5
goal_xy_weight=3.0
smooth_weight=0.75
```

This avoids PyCUDA-vs-Torch backend mixing in the official risk-aware ablation.

## Maps

- `config/b2_omni_oracle_low_friction_patch.yaml`: strong-claim candidate.
- `config/b2_omni_oracle_safe_corridor.yaml`: supporting evidence.
- `config/b2_omni_oracle_risk_band.yaml`: stress test / limitation.
- `config/b2_omni_oracle.yaml`: fixed two-obstacle standard visual benchmark and summary.

## Risk Weight Sweep

Official selection is based on a 10-episode full-config Torch sweep:

```bash
python3 tools/sweep_stage5_e_risk_cost.py \
  --configs <map.yaml> \
  --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_<map> \
  --episodes 10 \
  --base-seed 123 \
  --backend torch \
  --controllers nominal,learned \
  --risk-weights 0,0.5,1,3,5,10 \
  --risk-power 2.0 \
  --risk-threshold 0.3 \
  --risk-mode excess \
  --fdm-device cuda \
  --fdm-residual-gain 0.5 \
  --learned-goal-xy-weight 3.0 \
  --learned-smooth-weight 0.75
```

Selected official weights:

- low_friction_patch: `10`
- safe_corridor: `0.5`
- risk_band: `5` as stress-test setting
- two_obstacle_standard: `3`

## Official 50-Episode Benchmark

Run one sweep per map with risk weights `0,<selected>` and `50` episodes. Seed mapping is:

```text
seed = 123 + episode_id
episode_id = 0..49
```

Primary outputs:

- `stage5_benchmark_summary.json`
- `analysis/stage5_e_paired_stats.json`
- `analysis/stage5_e_paired_stats.csv`
- `figures/stage5_e/`
- `tables/stage5_e/`

## Statistics

Core metrics:

- `final_distance`
- `steps`
- `cumulative_terrain_risk`
- `max_terrain_risk`
- `terrain_risk_excess`
- `terrain_risk_exposure_ratio`
- `control_smoothness`
- `control_jerk`
- `mean_mppi_time_ms`

For lower-is-better metrics, `learned - nominal < 0` means learned is better. If a bootstrap CI crosses zero, do not call it a significant improvement.

## Result Boundary

Allowed claims:

- explicit risk cost changes planner behavior on selected maps;
- low_friction_patch is the strongest risk-aware evidence;
- safe_corridor supports the trend with weaker excess-risk effect;
- risk_band remains a stress test and limitation;
- fixed two-obstacle scene provides continuity and visual evidence.

Disallowed claims:

- learned-FDM-MPPI dominates nominal on all maps and metrics;
- risk_band is a clean success;
- closed-loop trajectories are GT;
- learned runtime is real-time equivalent to nominal.
