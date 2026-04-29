# Evaluation Protocol

Last updated: 2026-04-29

## Required Experiment Groups

1. Nominal-MPPI in nominal world.
2. Nominal-MPPI in oracle world.
3. Oracle-MPPI upper bound.
4. Learned-FDM prediction.
5. Learned-FDM-MPPI in oracle world.
6. Learned-FDM-MPPI + risk.

## Metrics

- `success_rate`
- `final_distance`
- `path_length`
- `arrival_time`
- `mean_velocity`
- `control_smoothness`
- `tracking_error`
- `residual_error`
- `ADE@1s`, `ADE@2s`, `ADE@4s`
- `FDE@1s`, `FDE@2s`, `FDE@4s`
- `yaw_error`
- `risk_count`
- `min_obstacle_distance`
- `avg_mppi_time_ms`
- `max_mppi_time_ms`

## Per-Run Output Contract

Each experiment should save:

```text
results/<experiment_name>/
  config.yaml
  summary.json
  trajectory.csv
  controls.csv
  residuals.csv
  terrain.csv
  trajectory.png
  residual_plot.png
  cost_plot.png
```

## Minimum `summary.json`

```json
{
  "success": true,
  "final_distance": 0.0,
  "path_length": 0.0,
  "arrival_time": 0.0,
  "mean_velocity": 0.0,
  "mean_mppi_time_ms": 0.0,
  "max_mppi_time_ms": 0.0,
  "tracking_error": 0.0,
  "risk_count": 0
}
```

## Stage 0 Baseline Acceptance

- No fatal error.
- Final distance to goal `< 0.4 m`.
- Average MPPI computation time `< 20 ms`.
- Summary, CSV, and PNG outputs are saved.
- Animation failure does not affect the main flow.

