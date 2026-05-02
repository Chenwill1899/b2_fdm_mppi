# Stage 5-E Nature-Style Figure Rules

## Reference

The requested style reference is:

```text
https://github.com/Yuan1z0825/nature-skills.git
```

The repository was reachable during this stage. Its `nature-figure` guidance emphasizes publication-ready Matplotlib figures, white backgrounds, sans-serif fonts, editable vector output, multi-panel information hierarchy, and PNG previews at high DPI. This project does not install the external skill as a dependency; `tools/plot_stage5_e_risk_aware_results.py` implements the equivalent local conventions so the stage remains reproducible from this repository alone.

## Local Conventions

- Background: white figure and axes background.
- Font: sans-serif with `Arial`, `DejaVu Sans`, then `Liberation Sans` fallback.
- Vector text: `svg.fonttype = none` when SVG output is used; Stage 5-E outputs PDF plus PNG.
- Raster preview: PNG at 300 DPI.
- Paper output: PDF for editable publication layout.
- Panels: lower-case panel labels `a`, `b`, `c`, `d` placed at the upper-left of each subplot.
- Axes: no top/right spines; minimal grid use; compact tick labels.
- Legends: no box, concise controller names, no "GT" label for closed-loop execution.
- Colors:
  - nominal: grey / blue-grey
  - nominal + risk: blue
  - learned: orange
  - learned + risk: red-purple
  - high-risk terrain: warm sequential colormap with alpha around `0.35`
- Line width: trajectory and time-series lines use consistent medium widths.
- Markers: start is green circle, goal is purple star, obstacles are grey circles with dashed orange safety rings.

## Figure Naming

Stage 5-E figure scripts write to:

```text
figures/stage5_e/
```

Main filenames:

```text
fig_s5e_main_risk_aware_summary.pdf
fig_s5e_main_risk_aware_summary.png
fig_s5e_paired_delta_boxplots.pdf
fig_s5e_paired_delta_boxplots.png
fig_s5e_risk_pareto.pdf
fig_s5e_risk_pareto.png
fig_s5e_risk_timeseries.pdf
fig_s5e_risk_timeseries.png
fig_s5e_two_obstacle_trajectory_over_risk.pdf
fig_s5e_two_obstacle_trajectory_over_risk.png
fig_s5e_two_obstacle_animation.gif
fig_s5e_trajectory_over_risk_map.pdf
fig_s5e_trajectory_over_risk_map.png
fig_s5e_runtime.pdf
fig_s5e_runtime.png
```

Tables write to:

```text
tables/stage5_e/table_s5e_main_results.csv
tables/stage5_e/table_s5e_paired_stats.csv
```

## Closed-Loop Label Rule

Closed-loop comparison figures must not label a trajectory as `GT`. Each controller is executed in the oracle world and has its own realized trajectory. Use these labels:

- `Nominal-MPPI execution in oracle world`
- `Risk-aware Nominal-MPPI execution in oracle world`
- `Learned-FDM-MPPI execution in oracle world`
- `Risk-aware Learned-FDM-MPPI execution in oracle world`

`GT` is reserved for open-loop replay against a recorded oracle trajectory.
