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

### 2026-04-29: Saved Pytest Report

- Goal: save test output artifacts for user verification.
- Command:

```bash
python3 -m pytest -q --junitxml=results/test_reports/20260429_221105/pytest.xml
```

- Result: `12 passed in 2.10s`.
- Local artifacts:
  - `results/test_reports/20260429_221105/pytest.log`
  - `results/test_reports/20260429_221105/pytest.xml`
- Note: `results/` is ignored by Git, so these artifacts are local verification outputs rather than versioned source files.

### 2026-04-29: Saved GIF Smoke Run

- Goal: save an actual `animation.gif` artifact for user visual inspection.
- Run type: CPU smoke run with injected controller, not a real PyCUDA MPPI performance run.
- Local result directory:

```text
results/sim_results/2026-04-29_22-12-26/
```

- Key artifacts:
  - `results/sim_results/2026-04-29_22-12-26/animation.gif` (`118352` bytes)
  - `results/sim_results/2026-04-29_22-12-26/test_summary.yaml`
  - `results/sim_results/2026-04-29_22-12-26/results.csv`
  - `results/sim_results/2026-04-29_22-12-26/path.png`
- Summary:
  - `steps: 20`
  - `failed: false`
  - `final_distance: 8.990005493164062`
  - `path_length: 1.0000001192092896`
  - `mean_mppi_time_ms: 0.028455257415771484`
- Note: this run verifies result saving and GIF generation. It is not a Stage 0 baseline acceptance run because it uses an injected smoke controller instead of the real MPPI controller.

### 2026-04-29: S0-003/S0-004 Short-Goal Baseline

- Goal: create and run a short-goal baseline config for Stage 0 tuning.
- Config:

```text
config/fdm_mppi_baseline_short.yaml
```

- Verification command:

```bash
python3 -m pytest -q
```

- Verification result: `13 passed in 2.10s`.
- Saved pytest report:
  - `results/test_reports/20260429_221616/pytest.log`
  - `results/test_reports/20260429_221616/pytest.xml`
- Real MPPI run command:

```bash
timeout 180s python3 - <<'PY'
from b2_fdm_mppi.config import load_config
from b2_fdm_mppi.simulation.runner import MppiSimulationRunner

config = load_config('config/fdm_mppi_baseline_short.yaml')
runner = MppiSimulationRunner(config)
summary = runner.run()
print(f'results_path={summary.results_path}')
print(f'steps={summary.steps}')
print(f'reached_goal={summary.reached_goal}')
print(f'failed={summary.failed}')
print(f'run_time={summary.run_time}')
print(f'animation={summary.results_path / "animation.gif"}')
PY
```

- Result directory:

```text
results/sim_results/2026-04-29_22-16-38/
```

- Key artifacts:
  - `results/sim_results/2026-04-29_22-16-38/animation.gif` (`116558` bytes)
  - `results/sim_results/2026-04-29_22-16-38/test_summary.yaml`
  - `results/sim_results/2026-04-29_22-16-38/results.csv`
  - `results/sim_results/2026-04-29_22-16-38/time_results.csv`
  - `results/sim_results/2026-04-29_22-16-38/path.png`
- Metrics:
  - `steps: 400`
  - `success: false`
  - `failed: false`
  - `final_distance: 1.2289246320724487`
  - `path_length: 7.043788433074951`
  - `run_time: 40.0`
  - `mean_mppi_time_ms: 1.1695504188537598`
  - `max_mppi_time_ms: 1.8770694732666016`
- Conclusion: compute time is well below the `<20 ms` threshold and result artifacts are saved, but Stage 0 acceptance fails because `final_distance` is greater than `0.4 m`. Next task is parameter tuning.

### 2026-04-29: S0-007 Straight Static Obstacle Scene

- Goal: test fixed map limits with target `[8, 0]` and one static obstacle at `[4, 0]`.
- Config:

```text
config/fdm_mppi_baseline_straight_obstacle.yaml
```

- Map limits:
  - `xlim: [-10, 10]`
  - `ylim: [-10, 10]`
- Pytest report:
  - `results/test_reports/20260429_221950/pytest.log`
  - `results/test_reports/20260429_221950/pytest.xml`
  - result: `15 passed in 1.05s`
- Real MPPI result directory:

```text
results/sim_results/2026-04-29_22-20-03/
```

- Key artifacts:
  - `results/sim_results/2026-04-29_22-20-03/animation.gif` (`308860` bytes)
  - `results/sim_results/2026-04-29_22-20-03/path.png`
  - `results/sim_results/2026-04-29_22-20-03/test_summary.yaml`
  - `results/sim_results/2026-04-29_22-20-03/results.csv`
  - `results/sim_results/2026-04-29_22-20-03/obs_results.csv`
  - `results/sim_results/2026-04-29_22-20-03/time_results.csv`
- Metrics:
  - `success: true`
  - `failed: false`
  - `steps: 346`
  - `final_distance: 0.37658464908599854`
  - `arrival_time: 34.6`
  - `path_length: 22.91684341430664`
  - `mean_mppi_time_ms: 1.193389038130038`
  - `max_mppi_time_ms: 1.7497539520263672`
  - `min_center_dist_to_obstacle: 1.1954317267638908`
  - `min_clearance_to_obstacle_surface_minus_robot_radius: 0.19543172676389076`
- Conclusion: target and compute-time Stage 0 criteria passed in this scene, but obstacle clearance is below `safety_dist=0.3 m`; next tuning should improve clearance.

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
