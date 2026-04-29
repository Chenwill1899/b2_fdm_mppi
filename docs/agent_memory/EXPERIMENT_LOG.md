# Experiment Log

Last updated: 2026-04-29

## Startup Inspection

- Command: `git status --short --branch`
- Result: repository on branch `fdm`, tracking `origin/fdm`.
- Command: `find . -maxdepth 3 -type f | sort | head -200`
- Result: ROS 2 package layout with `b2_fdm_mppi/`, `config/`, `launch/`, `tests/`, existing `build/`, `install/`, `log/`, and `results/`-related code paths.

## Experiments

### 2026-04-29: S0-001 Animation Non-Fatal Test

- Goal: verify animation writer failures do not abort baseline simulation.
- Change type: bugfix/unit test, not an algorithm experiment.
- Command:

```bash
python3 -m pytest tests/test_runner.py::test_runner_continues_when_animation_fails -q
```

- RED result before fix: failed with `RuntimeError: animation writer unavailable` propagated from `utils.animate_simulation()`.
- GREEN result after fix: `1 passed`.
- Full regression command:

```bash
python3 -m pytest -q
```

- Full regression result: `9 passed in 2.08s`.

### 2026-04-29: S0-001 GIF Output Requirement

- Goal: keep `animation.gif` as a default stage artifact while preserving non-fatal animation failures.
- Commands:

```bash
python3 -m pytest tests/test_config.py::test_default_config_loads_required_groups tests/test_runner.py::test_runner_saves_gif_when_animation_is_enabled tests/test_runner.py::test_runner_continues_when_animation_fails -q
python3 -m pytest -q
```

- Result: targeted animation tests passed; full regression result `10 passed in 2.09s`.
- Decision: `config/fdm_mppi.yaml` keeps `results.enable_animation: true`; runner still catches animation exceptions so missing writers do not abort the simulation.

### 2026-04-29: S0-002 Summary Metrics And GIF Axis

- Goal: add Stage 0 baseline summary metrics and widen `animation.gif` y-axis for visual inspection.
- Commands:

```bash
python3 -m pytest tests/test_runner.py::test_runner_summary_contains_stage0_metrics tests/test_visualization.py::test_animation_axis_limits_keep_wide_y_view_for_goal_on_x_axis -q
python3 -m pytest -q
```

- Result: targeted tests passed; full regression result `12 passed in 2.10s`.
- Summary file: `test_summary.yaml` now contains `success`, `final_distance`, `path_length`, `arrival_time`, `run_time`, `mean_mppi_time_ms`, and `max_mppi_time_ms`.
- Visualization: animation y-axis is fixed to `[-10, 10]`; default x-axis remains `[-1, target_x + 1]`.

## Next Baseline Experiment

Planned command:

```bash
python3 -m pytest -q
```

Then, if CUDA/PyCUDA runtime is available:

```bash
ros2 launch b2_fdm_mppi fdm_mppi.launch.py
```

Expected Stage 0 outputs:

- `results/sim_results/<timestamp>/results.csv`
- `results/sim_results/<timestamp>/obs_results.csv`
- `results/sim_results/<timestamp>/time_results.csv`
- `results/sim_results/<timestamp>/test_summary.yaml`
- Plot images when plotting is enabled.
