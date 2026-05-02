# Stage 5-E / S6 Risk-Aware Learned-FDM-MPPI Protocol

## Why Stage 5-D Is Not Enough

Stage 5-D showed that learned-FDM-MPPI can form calibrated operating modes:

- default `residual_gain=1.0`: conservative and smoother, but slower to the goal;
- efficiency `0.5/3.5/1.0`: better final distance and steps, with small risk and smoothness costs;
- balanced `0.5/3.0/0.75`: small step improvement with final distance and terrain risk close to nominal.

The limitation is methodological: terrain risk was an FDM input feature and an evaluation metric, but it was not an explicit MPPI planning objective. Therefore Stage 5-D cannot claim that learned-FDM-MPPI actively avoids high-risk terrain.

## Risk-Aware MPPI Cost

Stage 5-E promotes terrain risk into the planner objective. The MPPI cost can include:

```text
terrain_risk_weight * sum(phi(risk(x_t)))
```

with:

- `terrain_risk_mode=excess`
- `terrain_risk_threshold=0.3`
- `terrain_risk_power=2.0`

For excess mode:

```text
phi(risk) = max(risk - terrain_risk_threshold, 0) ** terrain_risk_power
```

The official analysis compares risk-off (`terrain_risk_weight=0`) and risk-on (`terrain_risk_weight>0`) under the same Torch rollout/cost backend.

## Same-Backend 2x2 Ablation

Official Stage 5-E conclusions must use `backend=torch` for both nominal and learned controllers:

1. nominal Torch, risk off
2. nominal Torch, risk on
3. learned Torch, risk off
4. learned Torch, risk on

The learned controller uses the balanced Stage 5-D candidate:

```text
residual_gain=0.5
goal_xy_weight=3.0
smooth_weight=0.75
```

This avoids a mixed PyCUDA nominal versus Torch learned comparison.

## Risk Maps

Official maps:

- `config/b2_omni_oracle_low_friction_patch.yaml`
- `config/b2_omni_oracle_safe_corridor.yaml`
- `config/b2_omni_oracle_risk_band.yaml`
- `config/b2_omni_oracle.yaml`

Optional:

- `config/b2_omni_oracle_risk_island.yaml`

Map-level interpretation:

- `low_friction_patch`: candidate strong risk-aware claim.
- `safe_corridor`: supporting claim.
- `risk_band`: stress test and possible limitation.
- `config/b2_omni_oracle.yaml`: fixed two-obstacle continuity and visual benchmark.
- `risk_island`: optional supporting/stress map if runtime permits.

## Risk Weight Selection

Run a 10-episode sweep before official 50-episode reporting:

```bash
python3 tools/sweep_stage5_e_risk_cost.py \
  --configs config/b2_omni_oracle_low_friction_patch.yaml,config/b2_omni_oracle_safe_corridor.yaml,config/b2_omni_oracle_risk_band.yaml,config/b2_omni_oracle.yaml \
  --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch \
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

Selection criteria:

- success rate does not fall below the risk-off baseline;
- cumulative risk, terrain-risk excess, or exposure ratio decreases;
- final distance and steps do not degrade unacceptably;
- obstacle clearance does not collapse;
- runtime is reported as an offline benchmark cost if high.

If a risk map only trades risk for much worse progress, report it as stress-test evidence.

## Official 50-Episode Benchmark

For each selected map/weight, run the same 2x2 structure with `50` episodes and `seed = 123 + episode_id`.

Required outputs per official case:

- `stage5_benchmark_summary.json`
- paired deltas in the benchmark summary;
- analysis JSON and CSV from `tools/analyze_stage5_e_risk_aware.py`;
- paper tables from `tools/plot_stage5_e_risk_aware_results.py`.

The fixed two-obstacle map must be included in the official summary, even if it is mainly used for visual continuity.

## Fixed Two-Obstacle Visual Benchmark

Use:

```text
config/b2_omni_oracle.yaml
```

Run the four visual cases:

- nominal Torch + risk off
- nominal Torch + risk on
- learned Torch balanced + risk off
- learned Torch balanced + risk on

Recommended command:

```bash
python3 tools/visualize_stage5_closed_loop.py \
  --config config/b2_omni_oracle.yaml \
  --scenario-name two_obstacle_standard \
  --output results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123 \
  --seed 123 \
  --backend torch \
  --fdm-device cuda \
  --fdm-residual-gain 0.5 \
  --risk-aware-2x2 \
  --risk-weight <selected_weight> \
  --risk-power 2.0 \
  --risk-threshold 0.3 \
  --risk-mode excess \
  --learned-goal-xy-weight 3.0 \
  --learned-smooth-weight 0.75
```

The plot tool then creates trajectory-over-risk maps, risk curves, cumulative-risk curves, and a copied learned-risk animation GIF.

## Statistics

Core paired metrics:

- `final_distance`
- `steps`
- `cumulative_terrain_risk`
- `max_terrain_risk`
- `terrain_risk_excess`
- `terrain_risk_exposure_ratio`
- `control_smoothness`
- `control_jerk`
- `mean_mppi_time_ms`

For lower-is-better metrics, `learned - nominal < 0` means learned is better. If a confidence interval crosses zero, describe the result as non-significant or mixed, not as a clear improvement.

## Result Boundary

Stage 5-E can support:

- explicit terrain-risk cost changes planner behavior;
- same-backend Torch learned-vs-nominal comparisons are fairer than PyCUDA-vs-Torch comparisons;
- map-wise claims for low-friction, safe-corridor, risk-band, and fixed two-obstacle scenes.

Stage 5-E must not claim:

- learned-FDM-MPPI dominates nominal on every map and metric;
- risk-band failure is a success;
- closed-loop oracle trajectories are ground truth;
- real-time parity with nominal Torch or nominal CUDA unless runtime evidence supports it.
