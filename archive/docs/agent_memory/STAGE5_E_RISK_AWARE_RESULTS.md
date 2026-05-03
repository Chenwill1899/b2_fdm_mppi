# Stage 5-E / S6 Risk-Aware Results

Date: 2026-05-02

## Scope

This package evaluates explicit terrain-risk cost for learned-FDM-MPPI with a same-backend Torch ablation. Stage 5-D treated terrain risk as an FDM input and post-hoc metric; Stage 5-E makes it an MPPI planning objective.

Official controller matrix:

- nominal Torch, risk off
- nominal Torch, risk on
- learned Torch balanced, risk off
- learned Torch balanced, risk on

Learned balanced setting:

```text
residual_gain=0.5
goal_xy_weight=3.0
smooth_weight=0.75
```

Risk objective:

```text
terrain_risk_mode=excess
terrain_risk_threshold=0.3
terrain_risk_power=2.0
```

## Commands

Risk-weight selection used 10 episodes per map:

```bash
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_low_friction_patch.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_low_friction --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_safe_corridor.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_safe_corridor --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_risk_band.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_risk_band --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_two_obstacle --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
```

Official 50-episode benchmarks:

```bash
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_low_friction_patch.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_safe_corridor.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_risk_band.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,5 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,3 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
```

Analysis and plotting:

```bash
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10/analysis
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5/analysis
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5/analysis
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3/analysis
python3 tools/visualize_stage5_closed_loop.py --config config/b2_omni_oracle.yaml --scenario-name two_obstacle_standard --output results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123 --seed 123 --backend torch --fdm-device cuda --fdm-residual-gain 0.5 --risk-aware-2x2 --risk-weight 3 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/plot_stage5_e_risk_aware_results.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3/stage5_e_risk_sweep_summary.json --visual-summary results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123/stage5_e_visual_eval_summary.json --output figures/stage5_e --tables-output tables/stage5_e
```

## Official Risk Weights

| Map | Official weight | Role |
| --- | ---: | --- |
| low_friction_patch | 10 | strong claim |
| safe_corridor | 0.5 | supporting claim |
| risk_band | 5 | stress test / limitation |
| two_obstacle_standard | 3 | fixed-scene visual continuity |

## Main 50-Episode Result

All rows are learned minus nominal within the same risk-weight case. Lower is better for all shown metrics except success-rate delta, where higher is better.

| Map | Weight | Success delta | Final delta | Cumulative risk delta | Excess risk delta | Exposure delta | Runtime delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_friction_patch | 10 | +0.06 | -0.0127 | -9.1263 | -3.7318 | -0.0204 | +21.57 ms |
| safe_corridor | 0.5 | +0.00 | -0.0019 | -2.2375 | +0.0040 | +0.0033 | +27.94 ms |
| risk_band | 5 | +0.00 | -0.0002 | -3.3511 | -0.3686 | +0.0005 | +21.08 ms |
| two_obstacle_standard | 3 | +0.04 | -0.0071 | -7.6334 | -2.7628 | +0.0139 | +17.65 ms |

## Paired Statistics

Selected risk-on comparisons, learned minus nominal:

| Map | Metric | Mean delta | 95% bootstrap CI | pct improved | Wilcoxon p |
| --- | --- | ---: | --- | ---: | ---: |
| low_friction_patch | final_distance | -0.0127 | [-0.0327, +0.0023] | 0.38 | 0.1407 |
| low_friction_patch | cumulative_terrain_risk | -9.1263 | [-15.6966, -4.0495] | 0.90 | 3.40e-11 |
| low_friction_patch | terrain_risk_excess | -3.7318 | [-6.1442, -1.8801] | 0.90 | 3.61e-12 |
| low_friction_patch | terrain_risk_exposure_ratio | -0.0204 | [-0.0314, -0.0105] | 0.74 | 4.35e-05 |
| safe_corridor | final_distance | -0.0019 | [-0.0045, +0.0007] | 0.56 | 0.2187 |
| safe_corridor | cumulative_terrain_risk | -2.2375 | [-2.9686, -1.5139] | 0.84 | 1.87e-07 |
| safe_corridor | terrain_risk_excess | +0.0040 | [-0.0369, +0.0458] | 0.54 | 0.8708 |
| risk_band | final_distance | -0.0002 | [-0.0024, +0.0020] | 0.46 | 0.8109 |
| risk_band | cumulative_terrain_risk | -3.3511 | [-4.4428, -2.3099] | 0.78 | 1.62e-07 |
| risk_band | terrain_risk_excess | -0.3686 | [-0.9771, +0.2267] | 0.58 | 0.3000 |
| two_obstacle_standard | final_distance | -0.0071 | [-0.0128, -0.0025] | 0.64 | 0.0035 |
| two_obstacle_standard | cumulative_terrain_risk | -7.6334 | [-10.2868, -4.7360] | 0.90 | 2.00e-07 |
| two_obstacle_standard | terrain_risk_excess | -2.7628 | [-3.5321, -1.9488] | 0.92 | 8.97e-09 |
| two_obstacle_standard | terrain_risk_exposure_ratio | +0.0139 | [-0.0029, +0.0384] | 0.50 | 0.8482 |

## Fixed Two-Obstacle Visual Benchmark

Visual scene:

```text
config/b2_omni_oracle.yaml
seed=123
backend=torch
terrain_risk_weight=3
```

| Controller | Success | Final | Steps | Cumulative risk | Excess risk | Exposure | Min clearance | Mean MPPI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal-MPPI execution in oracle world | true | 0.3476 | 245 | 105.50 | 35.58 | 0.8245 | 0.0976 | 10.18 ms |
| Risk-aware Nominal-MPPI execution in oracle world | true | 0.3365 | 223 | 97.28 | 33.22 | 0.8341 | 0.1041 | 10.25 ms |
| Learned-FDM-MPPI execution in oracle world | true | 0.3282 | 230 | 99.79 | 34.14 | 0.8391 | 0.0839 | 25.89 ms |
| Risk-aware Learned-FDM-MPPI execution in oracle world | true | 0.3394 | 220 | 94.31 | 31.74 | 0.7909 | 0.1023 | 26.53 ms |

Risk-aware learned has the lowest cumulative risk, excess risk, and exposure in this fixed visual scene. The figure labels intentionally avoid `GT`; each controller is a closed-loop execution in the oracle world.

## Figures and Tables

Tracked output:

- `figures/stage5_e/fig_s5e_main_risk_aware_summary.pdf/png`
- `figures/stage5_e/fig_s5e_paired_delta_boxplots.pdf/png`
- `figures/stage5_e/fig_s5e_risk_pareto.pdf/png`
- `figures/stage5_e/fig_s5e_risk_timeseries.pdf/png`
- `figures/stage5_e/fig_s5e_trajectory_over_risk_map.pdf/png`
- `figures/stage5_e/fig_s5e_two_obstacle_trajectory_over_risk.pdf/png`
- `figures/stage5_e/fig_s5e_two_obstacle_animation.gif`
- `figures/stage5_e/fig_s5e_runtime.pdf/png`
- `tables/stage5_e/table_s5e_main_results.csv`
- `tables/stage5_e/table_s5e_paired_stats.csv`

Raw run outputs remain under `results/stage5_e_risk_aware/` and are not tracked.

## Interpretation

- Strong result: low_friction_patch. Risk-aware learned reduces cumulative risk, excess risk, and exposure versus risk-aware nominal with CIs below zero. Final distance is better on average but its CI crosses zero, so it is not claimed as a statistically significant final-distance win.
- Supporting result: safe_corridor. Learned reduces cumulative risk and steps versus nominal; excess and exposure do not improve significantly under the selected weight.
- Stress test / limitation: risk_band. Learned reduces cumulative risk but does not clearly reduce excess risk or exposure, so this map should not be framed as a clean risk-aware success.
- Visual continuity: two_obstacle_standard. The fixed scene shows clear risk-map trajectory evidence and significant final/cumulative/excess improvements, but its risk exposure is not a significant learned-vs-nominal win in the 50-episode summary.
- Runtime boundary: learned Torch is consistently slower than nominal Torch by roughly `17.7-27.9 ms` per MPPI step in the official risk-on comparisons. These are offline benchmark results, not real-time equivalence claims.

## Validation

Final checks:

```bash
python3 -m pytest -q  # 173 passed in 30.92s
git diff --check      # passed
```
