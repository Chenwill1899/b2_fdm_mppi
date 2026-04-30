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

- Goal: test fixed map limits with target `[18, 0]` and one static obstacle at `[10, 0]`.
- Config:

```text
config/fdm_mppi_baseline_straight_obstacle.yaml
```

- Map limits:
  - `xlim: [0, 20]`
  - `ylim: [-10, 10]`
- Pytest report:
  - `results/test_reports/20260429_222401/pytest.log`
  - `results/test_reports/20260429_222401/pytest.xml`
  - result: `16 passed in 1.03s`
- Real MPPI result directory:

```text
results/sim_results/2026-04-29_22-24-13/
```

- Key artifacts:
  - `results/sim_results/2026-04-29_22-24-13/animation.gif` (`429871` bytes)
  - `results/sim_results/2026-04-29_22-24-13/path.png`
  - `results/sim_results/2026-04-29_22-24-13/test_summary.yaml`
  - `results/sim_results/2026-04-29_22-24-13/results.csv`
  - `results/sim_results/2026-04-29_22-24-13/obs_results.csv`
  - `results/sim_results/2026-04-29_22-24-13/time_results.csv`
- Metrics:
  - `success: true`
  - `failed: false`
  - `steps: 180`
  - `final_distance: 0.39142125844955444`
  - `arrival_time: 18.0`
  - `path_length: 18.364532470703125`
  - `mean_mppi_time_ms: 1.2182156244913738`
  - `max_mppi_time_ms: 1.8024444580078125`
  - `obstacle_x_minmax: [10.0, 10.0]`
  - `obstacle_y_minmax: [0.0, 0.0]`
  - `obstacle_dx_unique: [0.0]`
  - `obstacle_dy_unique: [0.0]`
  - `min_center_dist_to_obstacle: 1.2819332598406246`
  - `min_clearance_to_obstacle_surface_minus_robot_radius: 0.28193325984062456`
- Conclusion: target and compute-time Stage 0 criteria passed; obstacle is verified stationary. Clearance is still slightly below `safety_dist=0.3 m`, so next tuning should improve clearance.

### 2026-04-29: S0-008 Double Static Obstacle Scene

- Goal: test target `[18, 0]` with two stationary obstacles at `[6, 1]` and `[12, 1.5]`.
- Config:

```text
config/fdm_mppi_baseline_straight_obstacle.yaml
```

- Pytest report:
  - `results/test_reports/20260429_222723/pytest.log`
  - `results/test_reports/20260429_222723/pytest.xml`
  - result: `16 passed in 1.06s`
- Real MPPI result directory:

```text
results/sim_results/2026-04-29_22-27-37/
```

- Key artifacts:
  - `results/sim_results/2026-04-29_22-27-37/animation.gif` (`470769` bytes)
  - `results/sim_results/2026-04-29_22-27-37/path.png`
  - `results/sim_results/2026-04-29_22-27-37/test_summary.yaml`
  - `results/sim_results/2026-04-29_22-27-37/results.csv`
  - `results/sim_results/2026-04-29_22-27-37/obs_results.csv`
  - `results/sim_results/2026-04-29_22-27-37/time_results.csv`
- Metrics:
  - `success: true`
  - `failed: false`
  - `steps: 168`
  - `final_distance: 0.36382970213890076`
  - `arrival_time: 16.8`
  - `path_length: 17.662694931030273`
  - `mean_mppi_time_ms: 1.2644728024800618`
  - `max_mppi_time_ms: 1.735687255859375`
  - `obs0_position: [6.0, 1.0]`, `obs0_velocity: [0.0, 0.0]`
  - `obs1_position: [12.0, 1.5]`, `obs1_velocity: [0.0, 0.0]`
  - `obs0_min_clearance: 0.3174117563467942`
  - `obs1_min_clearance: 0.9285765019226636`
  - `min_clearance_all: 0.3174117563467942`
- Conclusion: Stage 0 straight double-obstacle scene meets target distance, compute-time, saved-artifact, and clearance criteria.

### 2026-04-29: S0-008 Harder Double Static Obstacle Scene

- Goal: test target `[18, 0]` with two stationary obstacles at `[6, 0.5]` and `[12, -1]`.
- Config:

```text
config/fdm_mppi_baseline_straight_obstacle.yaml
```

- Pytest report:
  - `results/test_reports/20260429_223033/pytest.log`
  - `results/test_reports/20260429_223033/pytest.xml`
  - result: `16 passed in 1.06s`
- Real MPPI result directory:

```text
results/sim_results/2026-04-29_22-30-49/
```

- Key artifacts:
  - `results/sim_results/2026-04-29_22-30-49/animation.gif` (`311376` bytes)
  - `results/sim_results/2026-04-29_22-30-49/path.png`
  - `results/sim_results/2026-04-29_22-30-49/test_summary.yaml`
  - `results/sim_results/2026-04-29_22-30-49/results.csv`
  - `results/sim_results/2026-04-29_22-30-49/obs_results.csv`
  - `results/sim_results/2026-04-29_22-30-49/time_results.csv`
- Metrics:
  - `success: false`
  - `failed: false`
  - `steps: 400`
  - `final_distance: 7.511897087097168`
  - `path_length: 11.018898010253906`
  - `mean_mppi_time_ms: 1.2540578842163086`
  - `max_mppi_time_ms: 1.7654895782470703`
  - `obs0_position: [6.0, 0.5]`, `obs0_velocity: [0.0, 0.0]`
  - `obs1_position: [12.0, -1.0]`, `obs1_velocity: [0.0, 0.0]`
  - `obs0_min_clearance: 0.2870425901433238`
  - `obs1_min_clearance: 0.44173382387842`
  - `min_clearance_all: 0.2870425901433238`
  - `final_xy: [10.544911, -0.9266779]`
- Conclusion: obstacles are stationary and artifacts are saved, but this harder layout fails Stage 0 target-distance and clearance criteria. Next task is tuning this scene.

### 2026-04-29: S1-001 B2 Omni Model Unit Tests

- Goal: add the nominal B2 omnidirectional SE(2) model before replacing the MPPI rollout model.
- Added:
  - `b2_fdm_mppi/core/omni_b2.py`
  - `config/b2_omni_nominal.yaml`
- Model:
  - state: `[x, y, theta, vx_real, vy_real, wz_real]`
  - control: `[vx_cmd, vy_cmd, wz_cmd]`
  - limits: `max_vx=1.5`, `max_vy=0.5`, `max_wz=1.0`
- Pytest report:
  - `results/test_reports/20260429_223738/pytest.log`
  - `results/test_reports/20260429_223738/pytest.xml`
  - result: `20 passed in 1.03s`
- Tests covered:
  - forward `vx` integration
  - lateral `vy` integration
  - yaw `wz` integration
  - body-frame velocity rotated into world frame
  - control clipping
  - 6D/3D config validation
- Conclusion: nominal B2 omni dynamics are validated in isolation. Next step is a separate NumPy omni MPPI controller; the existing CUDA controller remains differential-drive and should not be mutated in place.

### 2026-04-29: S1-003 NumPy Omni MPPI Controller

- Goal: add a CPU NumPy MPPI controller for the B2 omnidirectional nominal model before tuning the full scenario.
- Added:
  - `b2_fdm_mppi/controllers/mppi_omni_numpy.py`
  - `tests/test_mppi_omni_numpy.py`
- Controller behavior:
  - samples candidate control sequences with shape `[K, H, 3]`
  - clips controls to `vx=1.5`, `vy=0.5`, `wz=1.0`
  - rolls out candidates with `OmniB2`
  - scores goal, yaw, control, and obstacle clearance costs
  - performs MPPI weighted update and shifts the nominal control sequence
- Pytest report:
  - `results/test_reports/20260429_230911/pytest.log`
  - `results/test_reports/20260429_230911/pytest.xml`
  - result: `25 passed in 3.63s`
- Tests covered:
  - 3D control output and limits
  - forward motion toward an unobstructed goal
  - obstacle cost prefers lateral clearance
  - controller creation from `config/b2_omni_nominal.yaml`
  - closed-loop smoke motion with `OmniB2`
- Conclusion: NumPy omni MPPI core is validated in isolation. Next step is a scenario runner/logger to save CSV, PNG, GIF, and summary for the `[18,0]` double-obstacle scene.

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

### 2026-04-29: S1-004 Omni MPPI Runner and Tuned Double-Obstacle Scene

- Goal: run the B2 omni SE(2) nominal MPPI in the fixed map `x=[0,20]`, `y=[-10,10]` with target `[18,0]` and static obstacles `[6,0.5]`, `[12,-1]`.
- Added:
  - `b2_fdm_mppi/simulation/omni_runner.py`
  - `tools/run_omni_mppi.py`
  - `tests/test_omni_runner.py`
- Controller updates:
  - vectorized candidate rollout and obstacle cost
  - configurable `obstacle_weight`, `control_weight`, `smooth_weight`
  - smoothness cost includes the previous executed control and adjacent controls
- Visualization update:
  - `animation.gif` now draws sampled candidate rollouts in black and optimized rollout in orange.
- Tuned config:
  - `mppi.obstacle_weight: 800.0`
  - `mppi.control_weight: 0.01`
  - `mppi.smooth_weight: 1.0`
  - `robot.safety_dist: 0.4`
- Pytest report:
  - `results/test_reports/20260429_232717/pytest.log`
  - `results/test_reports/20260429_232717/pytest.xml`
  - result: `33 passed in 1.99s`
- Real omni MPPI result directory:

```text
results/sim_results/2026-04-29_23-27-26/
```

- Key artifacts:
  - `results/sim_results/2026-04-29_23-27-26/animation.gif` (`680K`)
  - `results/sim_results/2026-04-29_23-27-26/trajectory.png`
  - `results/sim_results/2026-04-29_23-27-26/summary.json`
  - `results/sim_results/2026-04-29_23-27-26/trajectory.csv`
  - `results/sim_results/2026-04-29_23-27-26/controls.csv`
- Metrics:
  - `success: true`
  - `steps: 134`
  - `final_distance: 0.36341118812561035`
  - `path_length: 18.1884765625`
  - `arrival_time: 13.4`
  - `mean_mppi_time_ms: 6.374088685903976`
  - `max_mppi_time_ms: 18.85843276977539`
  - `min_obstacle_clearance: 0.38778746128082275`
  - control mean absolute deltas: `vx=0.080398`, `vy=0.168291`, `wz=0.282473`
- Conclusion: Stage 1 omni nominal MPPI reaches the target in the harder double-obstacle scene, stays under the 20 ms mean compute target, saves summary/CSV/PNG/GIF, and restores candidate rollout visualization in GIF. Next stage after PR review is Oracle Residual World.

### 2026-04-29: S1-004 Result Directory Save Fix

- Issue: parameter tuning created many timestamped directories under `results/sim_results`, making it hard to identify the result the user should inspect.
- Fix:
  - Added `b2_fdm_mppi/simulation/results_path.py`.
  - Both legacy runner and omni runner now support `results.run_name` and `results.overwrite`.
  - `config/b2_omni_nominal.yaml` uses:

```yaml
results:
  root: "./results/sim_results"
  run_name: "b2_omni_nominal_latest"
  overwrite: true
```

- Formal run command:

```bash
python3 tools/run_omni_mppi.py --config config/b2_omni_nominal.yaml --seed 123
```

- Result directory:

```text
results/sim_results/b2_omni_nominal_latest/
```

- Key artifacts:
  - `animation.gif` (`695306` bytes)
  - `summary.json`
  - `trajectory.csv`
  - `controls.csv`
  - `trajectory.png`
- Metrics:
  - `success: true`
  - `final_distance: 0.36341118812561035`
  - `mean_mppi_time_ms: 5.523462793720302`
  - `max_mppi_time_ms: 8.809804916381836`
  - `min_obstacle_clearance: 0.38778746128082275`
- Pytest report:
  - `results/test_reports/20260429_233404/pytest.log`
  - `results/test_reports/20260429_233404/pytest.xml`
  - result: `34 passed in 1.97s`
- Conclusion: formal omni result output used one stable overwriteable directory for user inspection at this point. This was later changed on 2026-04-30 to timestamp-suffixed named directories.

### 2026-04-30: Named Timestamp Result Directories

- Goal: avoid overwriting the formal B2 omni result directory while keeping result names easy to identify.
- Change:
  - `create_results_path()` now supports `results.timestamp_suffix: true`.
  - `config/b2_omni_nominal.yaml` uses `run_name: b2_omni_nominal`, `timestamp_suffix: true`, and `overwrite: false`.
- Verification:
  - `python3 -m pytest -q`
  - result: `36 passed in 2.01s`
- Real run:
  - `results/sim_results/b2_omni_nominal_2026-04-30_13-43-14/`
  - `success: true`
  - `final_distance: 0.3889111578464508`
  - `mean_mppi_time_ms: 6.074447291237967`
  - `min_obstacle_clearance: 0.3978804349899292`

### 2026-04-30: CUDA B2 Omni MPPI With CBF Cost

- Goal: move the B2 omnidirectional MPPI rollout/cost evaluation onto CUDA first, then add CBF.
- Implementation:
  - Added `b2_fdm_mppi/controllers/mppi_omni_cuda.py`.
  - Added `mppi.backend` selection through `OmniMppiSimulationRunner`.
  - `tools/run_omni_mppi.py` now respects the configured backend.
  - Added discrete CBF penalty in the CUDA cost kernel:
    - `h = distance_to_obstacle - obstacle_radius - robot_radius - safety_dist`
    - violation uses `-(h_next - h + alpha * h)`
    - cost uses `mppi.cbf_weight * max(violation, 0)^2`
  - Current config uses `mppi.backend: cuda` and `mppi.cbf_weight: 500.0`.
- Tests added:
  - CUDA cost matches NumPy cost when `cbf_weight=0`.
  - CUDA CBF cost penalizes trajectories that decrease the barrier near an obstacle.
  - Runner selects CUDA backend when configured.
- Verification:
  - `python3 -m pytest -q`
  - result: `39 passed in 2.46s`
- Real run command:

```bash
python3 tools/run_omni_mppi.py --config config/b2_omni_nominal.yaml --seed 123
```

- Result directory:

```text
results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/
```

- Metrics:
  - `success: true`
  - `steps: 142`
  - `final_distance: 0.3754442036151886`
  - `path_length: 18.206396102905273`
  - `arrival_time: 14.200000000000001`
  - `mean_mppi_time_ms: 4.929683577846474`
  - `max_mppi_time_ms: 7.981300354003906`
  - `min_obstacle_clearance: 0.4295613765716553`
- Artifacts:
  - `animation.gif` (`665K`)
  - `trajectory.png` (`57K`)
  - `summary.json`
  - `trajectory.csv`
  - `controls.csv`
- Conclusion: the current Stage 1 runtime is CUDA-backed B2 omni MPPI with a CBF penalty cost. It is not a full soft/slack RCBF implementation yet.

### 2026-04-30: RCBF-Style Barrier Integration and Tuning

- Goal: replace the plain CUDA CBF penalty with the old project's RCBF-style relative-velocity barrier modes and tune the Stage 1 B2 omni scene.
- Implementation:
  - Extended `MppiOmniCuda` with `cbf.type` and `cbf.atau`.
  - Ported old barrier modes into the omni CUDA kernel:
    - `type=1`: cosine relative-velocity lookahead barrier (`h_csx` style).
    - `type=2`: direct relative-velocity lookahead barrier (`h_ex` style).
    - `type=3`: distance-constraint barrier.
  - The default config remains `cbf.type: 1`.
  - Added regression test proving an approaching obstacle is penalized more than an equally distant static obstacle.
- Tuning decision:
  - `num_trajectories=4096`: success, high safety, but mean runtime rose to about `12 ms`.
  - `num_trajectories=2048`: success, better runtime, but still not materially better than 1024.
  - `num_trajectories=1024` plus `minimum_distance=0.45`: best balance for this scene.
- Formal run:

```bash
python3 tools/run_omni_mppi.py --config config/b2_omni_nominal.yaml --seed 123
```

- Result directory:

```text
results/sim_results/b2_omni_nominal_2026-04-30_14-14-06/
```

- Metrics:
  - `success: true`
  - `steps: 144`
  - `final_distance: 0.34110841155052185`
  - `path_length: 18.318750381469727`
  - `arrival_time: 14.4`
  - `mean_mppi_time_ms: 4.540036122004191`
  - `max_mppi_time_ms: 11.853933334350586`
  - `min_obstacle_clearance: 0.4714846611022949`
- Verification:
  - `python3 -m pytest -q`
  - result: `40 passed in 2.02s`
- Conclusion: Stage 1 now has a tuned CUDA B2 omni MPPI controller with RCBF-style relative-velocity barrier cost. This is still a cost-based barrier, not the old soft/slack RCBF optimizer.

### 2026-04-30: Stage 1.5 Static-Obstacle Smoothness Baseline

- Goal: clean the Stage 1 B2 omni nominal MPPI baseline for static-obstacle FDM comparisons before entering oracle residual world work.
- Rationale:
  - The RCBF-style relative-velocity barrier is useful for dynamic obstacles.
  - Stage 1.5 is a static-obstacle baseline, so the default planner now relies on static obstacle distance penalty and keeps RCBF as an optional extension.
- Changes:
  - `config/b2_omni_nominal.yaml` default:
    - `mppi.cbf_weight: 0.0`
    - `cbf.enabled: false`
    - `cbf.type: 0`
    - `mppi.obstacle_weight: 300.0`
    - `robot.safety_dist: 0.5`
    - `execution.filter_enabled: true`
    - `execution.filter_alpha: 0.6`
  - Summary metrics now include:
    - `control_smoothness`
    - `smooth_vx`, `smooth_vy`, `smooth_wz`
    - `control_jerk`
    - `jerk_vx`, `jerk_vy`, `jerk_wz`
    - `vx_variance`, `vy_variance`, `wz_variance`
  - `controls.csv` stores executed controls.
  - `raw_controls.csv` stores raw MPPI commands before the execution low-pass filter.
  - `MppiOmniCuda.from_config()` forces `cbf_weight=0.0` when `cbf.enabled: false`.
- Old Stage 1 reference:
  - Result: `results/sim_results/b2_omni_nominal_2026-04-30_14-45-30/`
  - `success: true`
  - `final_distance: 0.34110841155052185`
  - `path_length: 18.318750381469727`
  - `mean_mppi_time_ms: 4.219803545210096`
  - `max_mppi_time_ms: 9.484291076660156`
  - `min_obstacle_clearance: 0.4714846611022949`
  - summary smoothness fields: not available
  - computed from `controls.csv`: `control_smoothness=0.1314503344222518`, `control_jerk=0.36088919826191274`
- New Stage 1.5 formal run:
  - Result: `results/sim_results/b2_omni_nominal_2026-04-30_14-52-40/`
  - `success: true`
  - `final_distance: 0.3399098217487335`
  - `path_length: 18.460805892944336`
  - `arrival_time: 16.2`
  - `mean_mppi_time_ms: 5.112684803244508`
  - `max_mppi_time_ms: 16.221046447753906`
  - `min_obstacle_clearance: 0.4645106792449951`
  - `control_smoothness: 0.012281207671864226`
  - `control_jerk: 0.020258904777513565`
  - `vx_variance: 0.1710001605578116`
  - `vy_variance: 0.018017247619684575`
  - `wz_variance: 0.03171373441428392`
- Verification:
  - `python3 -m pytest -q`
  - result: `43 passed in 2.56s`
- Conclusion: Stage 1.5 preserves success, final distance, runtime, and clearance while reducing computed control smoothness by about `90.7%` and control jerk by about `94.4%`. No FDM or oracle residual world changes were made.

### 2026-04-30: Stage 1.5 Pure SE2 vs Kinodynamic Baselines

- Goal: finalize Stage 1.5 nominal MPPI controls before Stage 2 oracle residual world work.
- Scope exclusions: no oracle residual world, no FDM dataset, no FDM training, no FDM-MPPI integration.
- Changes:
  - Added `config/b2_omni_pure_se2.yaml` as the weak pure SE(2)-MPPI baseline.
  - Added `config/b2_omni_kinodynamic.yaml` as the main kinodynamic nominal baseline.
  - Both configs keep `cbf.enabled=false`, `mppi.cbf_weight=0.0`, and execution filtering disabled.
  - Added far-field static obstacle potential with `mppi.obstacle_soft_weight` and `mppi.obstacle_influence_dist`.
  - Added sampling coverage summary metrics:
    - `sample_terminal_y_std_mean`
    - `sample_terminal_y_range_mean`
    - `sample_terminal_spread_mean`
    - `sample_terminal_x_range_mean`
  - Animation sampled and optimized rollouts now use the same kinodynamic rollout response model.
- Verification:
  - `python3 -m pytest -q`
  - result: `52 passed in 2.18s`
- Pure SE2 run:
  - Command: `python3 tools/run_omni_mppi.py --config config/b2_omni_pure_se2.yaml --seed 123`
  - Result: `results/sim_results/b2_omni_pure_se2_2026-04-30_16-26-22/`
  - `success: true`
  - `final_distance: 0.34273529052734375`
  - `min_obstacle_clearance: 0.48917269706726074`
  - `mean_mppi_time_ms: 4.515667484231191`
  - `control_smoothness: 0.1382118749747343`
  - `control_jerk: 0.3336636216416364`
  - `sample_terminal_y_std_mean: 0.0908367551911449`
  - `sample_terminal_y_range_mean: 0.42170131142723233`
  - `sample_terminal_spread_mean: 0.11228468836481957`
  - `sample_terminal_x_range_mean: 0.3947257046421913`
- Kinodynamic run:
  - Command: `python3 tools/run_omni_mppi.py --config config/b2_omni_kinodynamic.yaml --seed 123`
  - Result: `results/sim_results/b2_omni_kinodynamic_2026-04-30_16-27-08/`
  - `success: true`
  - `final_distance: 0.3254147171974182`
  - `min_obstacle_clearance: 1.302788257598877`
  - `mean_mppi_time_ms: 4.605328225340519`
  - `control_smoothness: 0.012360351873633395`
  - `control_jerk: 0.015468352090701201`
  - `acceleration_cost: 234.84668559903452`
  - `lateral_usage: 0.048483854998728086`
  - `yaw_rate_usage: 0.04249369600847192`
  - `sample_terminal_y_std_mean: 0.1175942634262934`
  - `sample_terminal_y_range_mean: 0.5386991505035692`
  - `sample_terminal_spread_mean: 0.15318969119647644`
  - `sample_terminal_x_range_mean: 0.5618406619232986`
- Conclusion: Kinodynamic remains within acceptance thresholds, removes execution-side double low-pass filtering, improves smoothness over Pure SE2, and increases sampled terminal spread for less concentrated prediction rollouts.

### 2026-04-30: Stage 1.5 Jitter and Over-Conservative Clearance Retune

- User observation: Kinodynamic trajectory looked shaky and stayed visually too far outside obstacle red safety circles.
- Diagnosis:
  - Prior Kinodynamic run `results/sim_results/b2_omni_kinodynamic_2026-04-30_16-27-08/` had `min_obstacle_clearance=1.3028 m`.
  - Red safety circle boundary corresponds to `clearance ~= safety_dist = 0.5 m`, so the trajectory was about `0.80 m` outside the red circle.
  - Control was acceleration-limited but still had high-frequency sign changes, so a rollout-internal jerk cost was added as an available tuning term.
- Final retuned run:
  - Result: `results/sim_results/b2_omni_kinodynamic_2026-04-30_16-44-27/`
  - `success: true`
  - `final_distance: 0.3299633860588074`
  - `path_length: 18.17268180847168`
  - `mean_mppi_time_ms: 4.433298394793556`
  - `max_mppi_time_ms: 8.055686950683594`
  - `min_obstacle_clearance: 0.4667545557022095`
  - `control_smoothness: 0.008712225537449353`
  - `control_jerk: 0.010106915998271083`
  - `acceleration_cost: 145.49416647540423`
  - `sample_terminal_y_range_mean: 0.4946968513845821`
  - `sample_terminal_spread_mean: 0.1497972121028426`
- Config outcome:
  - `obstacle_soft_weight=0.5`
  - `obstacle_influence_dist=1.2`
  - `max_vy=0.4`
  - `max_wz=0.7`
  - `max_ay=0.35`
  - `max_awz=0.8`
  - `execution.filter_enabled=false`
  - `cbf.enabled=false`
- Verification:
  - `python3 -m pytest -q`
  - result: `53 passed in 2.16s`
- Conclusion: Trajectory now runs close to the red safety boundary while preserving the Stage 1.5 acceptance thresholds and reducing visible smoothness/jerk metrics versus the prior kinodynamic baseline.
