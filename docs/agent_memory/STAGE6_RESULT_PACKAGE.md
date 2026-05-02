# Stage 6 Result Package: Stage 5-D Paper-Ready Results

## Scope

S6-001 packages the S5-010 50-episode ID/OOD benchmark into paper-ready figures, tables, and a compressed archive. It does not add a new controller or tune new parameters.

Source result:

```text
results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json
```

Official matrix:

- ID random, OOD obstacle, OOD terrain
- nominal CUDA
- default learned `residual_gain=1.0`
- efficiency learned `residual_gain=0.5`, `goal_xy_weight=3.5`, `smooth_weight=1.0`
- balanced learned `residual_gain=0.5`, `goal_xy_weight=3.0`, `smooth_weight=0.75`
- `50` episodes per controller/scenario

## Generated Outputs

Figures:

```text
figures/stage5
```

Tables:

```text
tables/stage5
```

Zip package:

```text
results/stage6_result_package/stage5_result_package.zip
```

## Main Result Framing

S5-010 supports multiple calibrated learned-FDM-MPPI operating modes, not a single learned controller that dominates every metric.

| controller | delta_final | delta_steps | delta_risk | delta_smooth | delta_jerk | delta_runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Default learned | 0.007983 | 24.426667 | -0.021790 | -0.000409 | -0.000811 | 48.912255 |
| Efficiency | -0.003217 | -7.153333 | 0.002467 | 0.000242 | 0.000107 | 49.653648 |
| Balanced | 0.000047 | -3.200000 | -0.000024 | 0.000106 | 0.000100 | 48.431096 |


## Figure Inventory

- `stage5_main_result_bars.png`: mean metric comparison across scenarios and controllers.
- `stage5_paired_delta_boxplots.png`: per-episode learned-minus-nominal deltas for final distance, steps, clearance, terrain risk, smoothness, jerk, and runtime.
- `stage5_pareto_scatter.png`: steps-vs-terrain-risk Pareto view.
- `stage5_trajectory_gallery.png`: representative paired trajectories for the three scenarios. The dashed circle around each goal is `simulation.minimum_distance`, i.e. the arrival tolerance.
- `stage5_runtime_bars.png`: runtime comparison.
- `stage5_failure_tradeoff_analysis.png`: compact trade-off summary.

## Table Inventory

- `stage5_main_results.csv/md`
- `stage5_operating_modes.csv/md`
- `stage5_paired_delta_summary.csv/md`
- `stage5_paired_episode_deltas.csv`
- `stage5_runtime.csv/md`

## Interpretation

- Default learned `g=1.0` is the conservative/smooth mode. It reduces terrain risk, smoothness, and jerk, but regresses final distance and steps.
- Efficiency `0.5/3.5/1.0` is the strongest official efficiency operating point. It improves final distance and steps most, with small risk/smoothness/jerk penalties.
- Balanced `0.5/3.0/0.75` is a balanced operating-point candidate. It modestly improves steps while keeping final distance and terrain risk close to nominal.
- Learned runtime remains much slower than nominal CUDA, so runtime profiling remains a separate Stage 6/Stage 7 requirement before real-time claims.

## Reproduction

```bash
python3 tools/plot_stage5_results.py
```
