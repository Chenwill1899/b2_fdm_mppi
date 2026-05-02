# Stage 5-E Nature-Style Figure Rules

Reference requested by the stage:

```text
https://github.com/Yuan1z0825/nature-skills.git
```

The repository was reachable during this stage. The local plotting tool does not install that external skill; instead `tools/plot_stage5_e_risk_aware_results.py` implements equivalent conventions in-repo so the paper figures are reproducible.

## Local Rules

- White background.
- Sans-serif font with `Arial`, `DejaVu Sans`, `Liberation Sans` fallback.
- High-DPI PNG preview and PDF paper output.
- Lower-case panel labels: `a`, `b`, `c`, `d`.
- Minimal axes, no top/right spines, no heavy grid.
- Color-blind-friendly controller/map colors.
- Consistent legend, line width, marker size, and compact tick labels.
- Terrain risk uses a warm colormap with transparent overlay.
- Closed-loop labels never use `GT`.

Closed-loop legend text:

- `Nominal-MPPI execution in oracle world`
- `Risk-aware Nominal-MPPI execution in oracle world`
- `Learned-FDM-MPPI execution in oracle world`
- `Risk-aware Learned-FDM-MPPI execution in oracle world`

## Outputs

Figures:

```text
figures/stage5_e/
```

Tables:

```text
tables/stage5_e/
```

Primary filenames:

```text
fig_s5e_main_risk_aware_summary.pdf/png
fig_s5e_paired_delta_boxplots.pdf/png
fig_s5e_risk_pareto.pdf/png
fig_s5e_risk_timeseries.pdf/png
fig_s5e_two_obstacle_trajectory_over_risk.pdf/png
fig_s5e_two_obstacle_animation.gif
fig_s5e_trajectory_over_risk_map.pdf/png
fig_s5e_runtime.pdf/png
table_s5e_main_results.csv
table_s5e_paired_stats.csv
```
