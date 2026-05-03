# Reproducibility Commands

Date: 2026-05-03

## Environment

Run from repository root:

```bash
cd /home/mexxiie/prj/py-mppi
```

Recommended validation:

```bash
python3 -m pytest -q
git diff --check
```

CUDA is used when available. CPU fallback is useful for smoke tests, but official Stage 5-E and Stage 6 runtime numbers were produced with Torch CUDA.

Raw `results/`, datasets, checkpoints, and model exports should not be committed.

## Stage 5-E Risk-Aware Official Runs

Risk-weight selection used 10 episodes per map:

```bash
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_low_friction_patch.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_low_friction --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_safe_corridor.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_safe_corridor --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_risk_band.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_risk_band --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle.yaml --output results/stage5_e_risk_aware/s5_e5_weight_sweep_10ep_full_torch_two_obstacle --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
```

Official 50-episode runs:

```bash
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_low_friction_patch.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_safe_corridor.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle_risk_band.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,5 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/sweep_stage5_e_risk_cost.py --configs config/b2_omni_oracle.yaml --output results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3 --episodes 50 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,3 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
```

## Stage 5-E Analysis And Figures

```bash
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10/analysis
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5/analysis
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5/analysis
python3 tools/analyze_stage5_e_risk_aware.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3/stage5_e_risk_sweep_summary.json --output results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3/analysis
python3 tools/visualize_stage5_closed_loop.py --config config/b2_omni_oracle.yaml --scenario-name two_obstacle_standard --output results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123 --seed 123 --backend torch --fdm-device cuda --fdm-residual-gain 0.5 --risk-aware-2x2 --risk-weight 3 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75
python3 tools/plot_stage5_e_risk_aware_results.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3/stage5_e_risk_sweep_summary.json --visual-summary results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123/stage5_e_visual_eval_summary.json --output figures/stage5_e --tables-output tables/stage5_e
```

## Stage 6 Runtime Closure

```bash
python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed_closeout --output results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout --episodes 10 --steps 10 --base-seed 123 --backend torch --device cuda --risk-weight 3.0
python3 tools/plot_stage6_runtime_results.py --summary results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode/stage6_runtime_matrix_summary.json results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout/stage6_runtime_matrix_summary.json --output figures/stage6 --tables-output tables/stage6
```

The runtime profiler defaults to forced-step semantics:

```text
force_steps=true
simulation.disable_goal_termination=true
steps_semantics=forced_compute_control_calls
```

Expected closeout consistency:

```text
profile_call_consistency.consistent=true
profile_call_consistency.unique_total_calls=[10]
```

## Final Tracked Outputs

Risk-aware figures:

- `figures/stage5_e/fig_s5e_main_risk_aware_summary.pdf/png`
- `figures/stage5_e/fig_s5e_paired_delta_boxplots.pdf/png`
- `figures/stage5_e/fig_s5e_risk_pareto.pdf/png`
- `figures/stage5_e/fig_s5e_risk_timeseries.pdf/png`
- `figures/stage5_e/fig_s5e_trajectory_over_risk_map.pdf/png`
- `figures/stage5_e/fig_s5e_two_obstacle_trajectory_over_risk.pdf/png`
- `figures/stage5_e/fig_s5e_two_obstacle_animation.gif`
- `figures/stage5_e/fig_s5e_runtime.pdf/png`

Runtime figures:

- `figures/stage6/fig_stage6_runtime_summary.pdf/png`
- `figures/stage6/fig_stage6_runtime_breakdown.pdf/png`
- `figures/stage6/fig_stage6_runtime_delta_buckets.pdf/png`

Tables:

- `tables/stage5_e/table_s5e_main_results.csv`
- `tables/stage5_e/table_s5e_paired_stats.csv`
- `tables/stage6/table_stage6_runtime_summary.csv`
- `tables/stage6/table_stage6_runtime_paired_deltas.csv`
