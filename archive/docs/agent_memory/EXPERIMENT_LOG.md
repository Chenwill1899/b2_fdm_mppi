# Experiment Log

Last updated: 2026-05-01

## 2026-05-01: S5-003 Torch CUDA Learned Rollout and Stage 5-B Benchmark

- Goal: make learned-FDM closed-loop benchmarking practical and run the first standard/ID/OOD Stage 5-B comparison.
- Code changes:
  - `b2_fdm_mppi/controllers/mppi_omni_learned_torch.py`
  - `create_omni_controller()` now creates `LearnedFdmMppiOmniTorch` for `fdm.enabled=true` and `mppi.backend=cuda`.
  - `tools/benchmark_learned_fdm_mppi.py` accepts `--backend cuda` and defaults FDM device to `cuda` for CUDA benchmark CLI runs.
- Benchmark outputs:
  - `results/stage5_benchmark/standard_seed123_cuda/stage5_benchmark_summary.json`
  - `results/stage5_benchmark/id_random_tasks_seed123_cuda/stage5_benchmark_summary.json`
  - `results/stage5_benchmark/ood_obstacle_seed123_cuda/stage5_benchmark_summary.json`
  - `results/stage5_benchmark/ood_terrain_seed123_cuda/stage5_benchmark_summary.json`
- Core results:

| Scenario | Episodes | Nominal success | Learned success | Nominal final dist | Learned final dist | Nominal steps | Learned steps | Nominal mean ms | Learned mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 1 | 1.00 | 1.00 | 0.3461 | 0.3371 | 225.0 | 214.0 | 6.59 | 27.03 |
| ID random | 20 | 1.00 | 1.00 | 0.6795 | 0.6930 | 137.1 | 155.8 | 6.41 | 50.75 |
| OOD obstacle | 20 | 1.00 | 1.00 | 0.6768 | 0.6878 | 142.4 | 175.1 | 7.19 | 53.85 |
| OOD terrain | 20 | 1.00 | 1.00 | 0.6848 | 0.6900 | 141.9 | 153.7 | 6.43 | 51.39 |

- Learned-minus-nominal deltas:
  - standard: final distance `-0.0090 m`, steps `-11.0`, clearance `+0.0826 m`, mean MPPI `+20.44 ms`.
  - ID random: final distance `+0.0135 m`, steps `+18.7`, clearance `-0.0366 m`, mean MPPI `+44.34 ms`.
  - OOD obstacle: final distance `+0.0109 m`, steps `+32.8`, clearance `+0.0276 m`, mean MPPI `+46.67 ms`.
  - OOD terrain: final distance `+0.0051 m`, steps `+11.8`, clearance `-0.0645 m`, mean MPPI `+44.96 ms`.
- Conclusion:
  - Torch CUDA learned rollout reduces the learned standard runtime from NumPy's `~1203 ms/step` to `~27 ms/step`.
  - Learned-FDM-MPPI improves the single standard scene.
  - ID/OOD random tasks reach 100% success, but learned does not stably outperform nominal on final distance, steps, or clearance.
  - Learned consistently reduces terrain risk, command-real error, residual norm, smoothness, and jerk.
  - Next step should be Stage 5-C profiling/tuning and closed-loop cost calibration before history-conditioned FDM.

## 2026-05-01: S5-002 Stage 5 Closed-loop Benchmark Runner

- Goal: add the PR #18 benchmark tool and summary schema for paired closed-loop Nominal-MPPI vs Learned-FDM-MPPI evaluation, without CUDA changes or full ID/OOD benchmark claims.
- Tool:

```text
tools/benchmark_learned_fdm_mppi.py
```

- Standard command:

```bash
python3 tools/benchmark_learned_fdm_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --scenario-name standard \
  --output results/stage5_benchmark/standard_seed123 \
  --episodes 1 \
  --base-seed 123 \
  --backend numpy \
  --controllers nominal,learned \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-checkpoint best_model.pt \
  --fdm-normalization normalization.npz \
  --fdm-device cpu
```

- Output:
  - `stage5_benchmark_summary.json`
  - `runs/standard_episode_0000_nominal/`
  - `runs/standard_episode_0000_learned/`
- Summary schema:
  - `metadata`: command, argv, git metadata, config/backend/controller/seed/model artifact metadata.
  - `runs`: per-controller per-episode closed-loop metrics.
  - `aggregates`: per-controller success rate and mean/std metrics.
  - `paired_deltas`: learned-minus-nominal matched by `scenario + episode_id + seed`.
- Boundary: PR #18 provides the runner and protocol only. Full standard/ID/OOD benchmark results belong to PR #19. Runtime profiling and acceleration belong to PR #20.

## 2026-05-01: S5-001 Learned Residual Dynamics Wrapper and NumPy Closed-loop Smoke

- Goal: start Stage 5 with a minimal NumPy-only learned-FDM MPPI smoke path, without CUDA changes or large benchmark claims.
- Code additions:
  - `b2_fdm_mppi/core/residual_fdm_model.py`
  - `b2_fdm_mppi/core/learned_residual_dynamics.py`
  - `b2_fdm_mppi/controllers/mppi_omni_learned_numpy.py`
  - FDM config/CLI overrides in `tools/run_omni_mppi.py`
  - `docs/agent_memory/STAGE5_PROTOCOL.md`
- Smoke commands:

```bash
python3 tools/run_omni_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --seed 123 \
  --backend numpy

python3 tools/run_omni_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --seed 123 \
  --backend numpy \
  --fdm-enabled \
  --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --fdm-checkpoint best_model.pt \
  --fdm-normalization normalization.npz \
  --fdm-device cpu
```

- Nominal result: `results/sim_results/b2_omni_oracle_2026-05-01_20-41-06`
  - reached_goal: `true`
  - failed: `false`
  - steps: `219`
  - final_distance: `0.3281706`
  - min_obstacle_clearance: `0.1524212`
  - mean_mppi_time_ms: `13.4969`
- Learned-FDM result: `results/sim_results/b2_omni_oracle_2026-05-01_20-42-31`
  - reached_goal: `true`
  - failed: `false`
  - steps: `209`
  - final_distance: `0.3429893`
  - min_obstacle_clearance: `0.1900530`
  - mean_mppi_time_ms: `1143.4956`
  - max_mppi_time_ms: `1224.5321`
- Acceptance:
  - learned final distance threshold: `0.4938048`
  - learned final distance: `0.3429893`
  - required CSV/JSON outputs exist.
- Known limitation: first NumPy learned rollout is stable but not real-time; runtime is dominated by Python terrain feature evaluation and Torch inference inside the MPPI horizon loop.
- Boundary: this is Stage 5 entry smoke only. It does not complete Stage 5 closed-loop benchmarking and does not validate CUDA learned rollout.

## 2026-05-01: S4-005 Multi-seed, OOD, and Stage 4 Protocol Closeout

- Goal: close Stage 4 residual FDM baseline validation without entering Stage 5 closed-loop MPPI integration.
- Code/docs additions:
  - `tools/evaluate_residual_fdm_dataset.py`
  - `config/b2_omni_oracle_random100_dataset_ood_obstacle.yaml`
  - `config/b2_omni_oracle_random100_dataset_ood_terrain.yaml`
  - `docs/agent_memory/STAGE4_PROTOCOL.md`
- Multi-seed training:
  - seeds: `123`, `456`, `789`
  - dataset: `datasets/oracle_stage4_splits`
  - output summary: `results/fdm_baselines/stage4_seed_benchmark_summary.json`
  - mean test MSE: `1.0148782469817283e-05`
  - std test MSE: `2.1833955071817022e-07`
  - mean test MSE reduction: `98.3862424492059%`
  - std test MSE reduction: `0.03471816443561322%`
  - mean test improvement: `61.99551928990127x`
- Best checkpoint standard open-loop eval:
  - command uses `--checkpoint best_model.pt`
  - output: `results/fdm_rollout_eval/stage4_mlp_seed123_hardened_b2_omni_oracle_seed123`
  - learned ADE/FDE: `0.04843650385737419` / `0.08230284601449966`
  - nominal ADE/FDE: `0.5212535262107849` / `0.9190490245819092`
  - residual MSE improvement: `98.53393274580017%`
  - compared with previous final `model.pt`: ADE slightly better, FDE substantially better, residual MSE slightly worse.
- OOD datasets:
  - OOD obstacle: 100/100 episodes, `13680` transitions; split validation pass.
  - OOD terrain: 100/100 episodes, `15017` transitions; split validation pass.
- OOD one-step residual eval:
  - OOD obstacle test MSE: `9.93157664197497e-06`; zero baseline `0.000568110088352114`; improvement `57.202406912014816x`.
  - OOD terrain test MSE: `1.0731993825174868e-05`; zero baseline `0.0006400654674507678`; improvement `59.640871759478344x`.
- OOD open-loop rollout eval:
  - OOD obstacle learned ADE/FDE: `0.020704902708530426` / `0.046311039477586746`; nominal ADE/FDE: `0.22822964191436768` / `0.32846060395240784`; residual MSE improvement `98.24939690291691%`.
  - OOD terrain learned ADE/FDE: `0.024650704115629196` / `0.05385246500372887`; nominal ADE/FDE: `0.26478996872901917` / `0.39450472593307495`; residual MSE improvement `98.27557548455931%`.
- Verification:

```bash
python3 -m pytest -q
```

- Result before OOD runs: `114 passed, 2 warnings in 20.02s`.
- Conclusion: Stage 4 ID and OOD open-loop evidence is stable enough to prepare Stage 5 planning, but Stage 5 is not yet implemented or validated.

## 2026-05-01: S4-004 Hardened Residual FDM Training Baseline

- Goal: make the Stage 4.1 one-step residual FDM baseline reproducible and checkpoint-trackable without changing model architecture or integrating learned FDM into MPPI.
- Tool:

```text
tools/train_residual_fdm.py
```

- Command:

```bash
python3 tools/train_residual_fdm.py \
  --dataset datasets/oracle_stage4_splits \
  --output results/fdm_baselines/stage4_mlp_seed123_hardened \
  --epochs 50 \
  --batch-size 512 \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --seed 123 \
  --device cpu
```

- Output directory:

```text
results/fdm_baselines/stage4_mlp_seed123_hardened/
```

- Required artifacts verified:
  - `model.pt`
  - `best_model.pt`
  - `normalization.npz`
  - `metrics.json`
  - `tensorboard/events.out.tfevents.*`
- Metrics:
  - `val_mse: 1.000058364297729e-05`
  - `test_mse: 1.0051174285763409e-05`
  - `zero_residual_val_mse: 0.0006534629501402378`
  - `zero_residual_test_mse: 0.0006288914009928703`
  - `val_mse_relative_improvement_pct: 98.46960204234516`
  - `test_mse_relative_improvement_pct: 98.40176312318867`
  - `val_mse_vx/vy/wz: 1.1322053978801705e-05 / 8.929766408982687e-06 / 9.749930541147478e-06`
  - `test_mse_vx/vy/wz: 1.1244998859183397e-05 / 8.813855856715236e-06 / 1.0094667231896892e-05`
  - `val_rmse_vx/vy/wz: 0.003364825995323043 / 0.0029882714751144493 / 0.0031224878768615705`
  - `test_rmse_vx/vy/wz: 0.0033533563573207364 / 0.0029688138804437095 / 0.0031772106055307212`
  - `val_mse_reduction_pct_vx/vy/wz: 99.16653595010273 / 94.74558556452628 / 97.743111778077`
  - `test_mse_reduction_pct_vx/vy/wz: 99.13072858537161 / 94.42857479590664 / 97.67866540006683`
  - `best_epoch: 46`
  - `best_val_loss: 0.028149016201496124`
  - `final_epoch: 50`
  - `final_val_loss: 0.02840588055551052`
- Reproducibility metadata:
  - `git_sha: a410272a1ab98132959bfceb17366ee0344d2ca7`
  - `git_branch: dev`
  - `git_dirty: true`
  - `device: cpu`
  - exact command written in `metrics.json`
  - `best_checkpoint_path: results/fdm_baselines/stage4_mlp_seed123_hardened/best_model.pt`
  - `final_checkpoint_path: results/fdm_baselines/stage4_mlp_seed123_hardened/model.pt`
- Verification:

```bash
python3 -m pytest -q
```

- Result: `112 passed, 2 warnings in 18.97s`.
- Conclusion: hardened Stage 4.1 training baseline remains strongly better than the zero-residual baseline while adding reproducibility metadata and best/final checkpoint tracking. This is still Stage 4.1; no Stage 5 closed-loop MPPI integration was added.

## 2026-05-01: S4-003 Open-loop Learned FDM Rollout Eval

- Goal: visualize and quantify trained residual FDM effects in the Stage 2 `config/b2_omni_oracle.yaml` oracle simulation environment without yet integrating learned FDM into MPPI rollout.
- Tool:

```text
tools/evaluate_residual_fdm_rollout.py
```

- Command:

```bash
python3 tools/evaluate_residual_fdm_rollout.py \
  --config config/b2_omni_oracle.yaml \
  --model-dir results/fdm_baselines/stage4_mlp_seed123 \
  --output results/fdm_rollout_eval/stage4_mlp_seed123_b2_omni_oracle_seed123 \
  --seed 123 \
  --backend numpy \
  --device cpu \
  --checkpoint model.pt \
  --normalization normalization.npz \
  --gif-fps 8 \
  --gif-max-frames 120
```

- Output directory:

```text
results/fdm_rollout_eval/stage4_mlp_seed123_b2_omni_oracle_seed123/
```

- Key artifacts:
  - `rollout_compare.gif`
  - `trajectory_compare.png`
  - `residual_compare.png`
  - `rollout_metrics.json`
  - `rollout_replay.npz`
  - `oracle_run/`
- Metrics:
  - `oracle_reached_goal: true`
  - `oracle_steps: 219`
  - `nominal_ade_xy: 0.5212535262107849`
  - `learned_ade_xy: 0.04870650917291641`
  - `nominal_fde_xy: 0.9190490245819092`
  - `learned_fde_xy: 0.14395728707313538`
  - `learned_vs_nominal_ade_improvement_pct: 90.65588879043469`
  - `nominal_ade_xy_at_1s: 0.019368547946214676`
  - `learned_ade_xy_at_1s: 0.0010764976032078266`
  - `nominal_fde_xy_at_1s: 0.03956378623843193`
  - `learned_fde_xy_at_1s: 0.0011031425092369318`
  - `nominal_ade_xy_at_2s: 0.04252150282263756`
  - `learned_ade_xy_at_2s: 0.0010103760287165642`
  - `nominal_fde_xy_at_2s: 0.09266608953475952`
  - `learned_fde_xy_at_2s: 0.0012467068154364824`
  - `nominal_ade_xy_at_4s: 0.09689775109291077`
  - `learned_ade_xy_at_4s: 0.00154116319026798`
  - `nominal_fde_xy_at_4s: 0.2138642817735672`
  - `learned_fde_xy_at_4s: 0.0027648615650832653`
  - `residual_mse: 1.0517556802369654e-05`
  - `zero_residual_mse: 0.0009893554961308837`
  - `residual_mse_improvement_pct: 98.93692845054167`
  - `rollout_compare.gif: 120 frames, 700x700, 977494 bytes`
- Reproducibility metadata now written to `rollout_metrics.json` / `.yaml`:
  - exact command
  - git SHA / branch / dirty flag
  - backend and device
  - checkpoint and normalization artifact paths
  - GIF enabled/fps/max-frame parameters
- Parameter snapshot:
  - `robot.radius: 0.6`
  - `robot.safety_dist: 0.25`
  - `obstacle_radii: [0.4, 0.4]`
  - `visualized_safety_boundary_radii: [1.25, 1.25]`
- Conclusion: the seed123 learned FDM replay substantially improves over nominal replay in the fixed Stage 2 oracle scene. This is an open-loop Stage 4 validation, not closed-loop learned-FDM MPPI.

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

### 2026-05-01: Stage 5 Closed-loop Visual Eval Standardization

- Goal: make the "learning 前后 closed-loop 效果怎么看" workflow repeatable instead of relying on manual ad hoc runs.
- Added:
  - `tools/visualize_stage5_closed_loop.py`
  - `docs/agent_memory/STAGE5_VISUAL_EVAL.md`
- Standard command:
  - `python3 tools/visualize_stage5_closed_loop.py --config config/b2_omni_oracle.yaml --scenario-name standard --output results/stage5_visual_eval/standard_seed123_cuda --seed 123 --backend cuda --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda`
- Outputs:
  - `stage5_visual_eval_summary.json`
  - `closed_loop_nominal_vs_learned.png`
  - `closed_loop_compare_metrics.csv`
  - `closed_loop_compare_metrics.json`
  - per-run `trajectory.png` and `animation.gif` for nominal and learned.
- Current manual visual check before tool standardization:
  - Nominal: `results/sim_results/b2_omni_oracle_2026-05-01_22-48-39`
  - Learned: `results/sim_results/b2_omni_oracle_2026-05-01_22-44-09`
  - Overlay: `results/stage5_visual_compare_seed123/closed_loop_nominal_vs_learned.png`
  - Learned vs nominal: final distance `0.3371` vs `0.3461`, steps `214` vs `225`, min clearance `0.2024` vs `0.1198`, mean terrain risk `0.4140` vs `0.4289`, mean MPPI time `27.73 ms` vs `7.18 ms`.
- Boundary: visual eval is single-scene qualitative inspection. Stage 5 improvement claims still require benchmark evidence from `tools/benchmark_learned_fdm_mppi.py`.

### 2026-05-01: Stage 5-C Residual Gain, Cost Calibration, And Runtime Profiling

- Goal: move from adding Stage 5 features to explaining why learned-FDM improves the standard scene but is not stable on ID/OOD random-task final distance and steps.
- PR chain cleanup:
  - PR #18 merged into `fdm` after fixing a brittle branch-name test.
  - PR #19 rebased onto the updated `fdm`, retargeted from `dev` to `fdm`, marked ready, validated, and merged.
- Code changes:
  - Added `fdm.residual_gain` to learned NumPy/Torch rollout.
  - Added `--fdm-residual-gain` to `tools/run_omni_mppi.py`, `tools/benchmark_learned_fdm_mppi.py`, and `tools/visualize_stage5_closed_loop.py`.
  - Added learned-controller-only MPPI cost overrides to the benchmark runner for:
    - `goal_xy_weight`
    - `obstacle_weight`
    - `obstacle_soft_weight`
    - `smooth_weight`
    - `accel_weight`
    - `lateral_weight`
    - `yaw_rate_weight`
  - Added `tools/sweep_stage5_calibration.py` for residual-gain and cost-grid sweeps.
  - Added `tools/profile_stage5_learned_torch.py` plus Torch controller runtime buckets:
    - `sample_candidates_ms`
    - `rollout_total_ms`
    - `terrain_features_ms`
    - `fdm_inference_ms`
    - `state_integrate_ms`
    - `obstacle_cost_ms`
    - `cost_terms_ms`
    - `update_distribution_ms`
    - `cpu_transfer_ms`
- Verification:
  - `python3 -m pytest tests/test_mppi_omni_learned_torch.py::test_learned_torch_rollout_scales_residual_with_gain tests/test_mppi_omni_learned_torch.py::test_learned_torch_profile_records_runtime_buckets tests/test_stage5_benchmark.py::test_parse_mppi_overrides_parses_supported_cost_keys tests/test_stage5_calibration_sweep.py::test_expand_sweep_cases_builds_residual_gain_and_cost_grid_product tests/test_stage5_runtime_profile.py::test_run_profile_enables_learned_torch_profiling_and_writes_summary -q`
  - result: `5 passed`
  - `python3 -m pytest tests/test_mppi_omni_learned_numpy.py tests/test_mppi_omni_learned_torch.py tests/test_omni_runner.py tests/test_stage5_benchmark.py tests/test_stage5_visual_eval.py tests/test_stage5_calibration_sweep.py tests/test_stage5_runtime_profile.py -q`
  - result after fixing NumPy config assertion: targeted Stage 5 suite passed locally.
  - `python3 -m pytest -q`
  - result: `147 passed in 20.27s`
  - `git diff --check`
  - result: passed
  - `python3 -m py_compile tools/sweep_stage5_calibration.py tools/profile_stage5_learned_torch.py tools/benchmark_learned_fdm_mppi.py tools/visualize_stage5_closed_loop.py`
- Standard residual-gain quick sweep:
  - Command: `python3 tools/sweep_stage5_calibration.py --config config/b2_omni_oracle.yaml --scenario-name standard_residual_gain_quick --output results/stage5_calibration/standard_residual_gain_seed123_cuda --episodes 1 --base-seed 123 --backend cuda --controllers nominal,learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --residual-gains 0.0,0.25,0.5,0.75,1.0`
  - Output: `results/stage5_calibration/standard_residual_gain_seed123_cuda/stage5_calibration_sweep_summary.json`
  - Learned results:
    - `gain=0.0`: success `1.00`, final distance `0.3476`, steps `245.0`, mean MPPI `26.28 ms`
    - `gain=0.25`: success `1.00`, final distance `0.3346`, steps `240.0`, mean MPPI `24.82 ms`
    - `gain=0.5`: success `1.00`, final distance `0.3439`, steps `217.0`, mean MPPI `25.27 ms`
    - `gain=0.75`: success `1.00`, final distance `0.3416`, steps `211.0`, mean MPPI `25.41 ms`
    - `gain=1.0`: success `1.00`, final distance `0.3371`, steps `214.0`, mean MPPI `25.11 ms`
- ID random residual-gain quick sweep:
  - Command: `python3 tools/sweep_stage5_calibration.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_tasks_residual_gain_quick --output results/stage5_calibration/id_random_tasks_residual_gain_seed123_cuda --episodes 5 --base-seed 123 --backend cuda --controllers nominal,learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --residual-gains 0.0,0.25,0.5,0.75,1.0`
  - Output: `results/stage5_calibration/id_random_tasks_residual_gain_seed123_cuda/stage5_calibration_sweep_summary.json`
  - Nominal reference in this sweep: success `1.00`, final distance `0.6827`, steps `125.8`
  - Learned results:
    - `gain=0.0`: success `1.00`, final distance `0.6826`, steps `115.2`, mean MPPI `49.89 ms`
    - `gain=0.25`: success `1.00`, final distance `0.6811`, steps `119.0`, mean MPPI `49.40 ms`
    - `gain=0.5`: success `1.00`, final distance `0.6811`, steps `121.2`, mean MPPI `49.94 ms`
    - `gain=0.75`: success `1.00`, final distance `0.6927`, steps `134.4`, mean MPPI `49.63 ms`
    - `gain=1.0`: success `1.00`, final distance `0.6926`, steps `147.0`, mean MPPI `49.29 ms`
- Cost sanity grid:
  - Command: `python3 tools/sweep_stage5_calibration.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_tasks_cost_quick --output results/stage5_calibration/id_random_tasks_cost_seed123_cuda --episodes 5 --base-seed 123 --backend cuda --controllers learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --residual-gains 0.5 --cost-grid goal_xy_weight=2.5,3.5 --cost-grid smooth_weight=0.5,1.0`
  - Output: `results/stage5_calibration/id_random_tasks_cost_seed123_cuda/stage5_calibration_sweep_summary.json`
  - Learned results:
    - `goal_xy_weight=2.5`, `smooth_weight=0.5`: final distance `0.6842`, steps `124.4`
    - `goal_xy_weight=2.5`, `smooth_weight=1.0`: final distance `0.6811`, steps `121.2`
    - `goal_xy_weight=3.5`, `smooth_weight=0.5`: final distance `0.6707`, steps `116.0`
    - `goal_xy_weight=3.5`, `smooth_weight=1.0`: final distance `0.6774`, steps `114.8`
- Runtime profile:
  - Command: `python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_random100_dataset.yaml --output results/stage5_profile/id_random_tasks_seed123_cuda_10steps --steps 10 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5`
  - Output: `results/stage5_profile/id_random_tasks_seed123_cuda_10steps/stage5_runtime_profile_summary.json`
  - Top profile buckets:
    - `rollout_total_ms`: total `587.55`, mean `58.76`, count `10`
    - `terrain_features_ms`: total `399.81`, mean `1.60`, count `250`
    - `fdm_inference_ms`: total `68.21`, mean `0.273`, count `250`
    - `state_integrate_ms`: total `68.29`, mean `0.273`, count `250`
    - `obstacle_cost_ms`: total `46.66`, mean `4.67`, count `10`
  - Note: profiling synchronizes timing buckets, so profiled `mean_mppi_time_ms` is higher than normal benchmark runtime. Use bucket proportions, not direct runtime, for bottleneck conclusions.
- Conclusion:
  - The ID random-task issue is not simply "MLP residual bad"; full residual correction appears too strong. Partial or zero residual gain removes the steps/final-distance degradation in the 5-episode ID quick sweep.
  - `residual_gain=0.5`, `goal_xy_weight=3.5`, and `smooth_weight=0.5/1.0` are the current Stage 5-C candidates for a 20-episode rerun.
  - The biggest runtime hotspot is terrain feature/risk computation inside the Torch rollout loop, followed by FDM inference/state integration and obstacle cost.

### 2026-05-02: S5-008 Calibrated 20-Episode ID/OOD Benchmark

- Goal: rerun the Stage 5-C candidates on full 20-episode ID/OOD suites after the quick residual-gain and cost sweeps.
- Seed mapping: `seed = base_seed + episode_id`, with `base_seed=123` and `episode_id=0..19`, so seeds are `123..142`.
- Backend/model:
  - backend: `cuda`
  - learned device: `cuda`
  - model dir: `results/fdm_baselines/stage4_mlp_seed123_hardened`
  - checkpoint: `best_model.pt`
  - normalization: `normalization.npz`
- Commands:
  - ID default paired rerun:
    - `python3 tools/benchmark_learned_fdm_mppi.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_tasks_default_20ep --output results/stage5_calibration/s5_008/id_random_default20_seed123_cuda --episodes 20 --base-seed 123 --backend cuda --controllers nominal,learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 1.0`
  - ID calibrated sweep:
    - `python3 tools/sweep_stage5_calibration.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_tasks_calibrated_20ep --output results/stage5_calibration/s5_008/id_random_calibrated20_seed123_cuda --episodes 20 --base-seed 123 --backend cuda --controllers learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --residual-gains 0.5 --cost-grid goal_xy_weight=3.5 --cost-grid smooth_weight=0.5,1.0`
  - OOD obstacle calibrated sweep:
    - `python3 tools/sweep_stage5_calibration.py --config config/b2_omni_oracle_random100_dataset_ood_obstacle.yaml --scenario-name ood_obstacle_calibrated_20ep --output results/stage5_calibration/s5_008/ood_obstacle_calibrated20_seed123_cuda --episodes 20 --base-seed 123 --backend cuda --controllers learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --residual-gains 0.5 --cost-grid goal_xy_weight=3.5 --cost-grid smooth_weight=0.5,1.0`
  - OOD terrain calibrated sweep:
    - `python3 tools/sweep_stage5_calibration.py --config config/b2_omni_oracle_random100_dataset_ood_terrain.yaml --scenario-name ood_terrain_calibrated_20ep --output results/stage5_calibration/s5_008/ood_terrain_calibrated20_seed123_cuda --episodes 20 --base-seed 123 --backend cuda --controllers learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --residual-gains 0.5 --cost-grid goal_xy_weight=3.5 --cost-grid smooth_weight=0.5,1.0`
- Output summaries:
  - `results/stage5_calibration/s5_008/id_random_default20_seed123_cuda/stage5_benchmark_summary.json`
  - `results/stage5_calibration/s5_008/id_random_calibrated20_seed123_cuda/stage5_calibration_sweep_summary.json`
  - `results/stage5_calibration/s5_008/ood_obstacle_calibrated20_seed123_cuda/stage5_calibration_sweep_summary.json`
  - `results/stage5_calibration/s5_008/ood_terrain_calibrated20_seed123_cuda/stage5_calibration_sweep_summary.json`
  - OOD default nominal/learned references are reused from the existing Stage 5-B 20-episode summaries under `results/stage5_benchmark/ood_obstacle_seed123_cuda/` and `results/stage5_benchmark/ood_terrain_seed123_cuda/`.
- Failure check:
  - all S5-008 generated `summary.json` files reported success; failed or unsuccessful runs: `0`.

| Scenario | Controller | Success | Final Dist | Delta Final | Steps | Delta Steps | Clearance | Terrain Risk | Smooth | Jerk | MPPI ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID random | nominal CUDA | 1.00 | 0.6795 | +0.0000 | 137.1 | +0.0 | 3.5621 | 0.3285 | 0.003186 | 0.002828 | 4.51 |
| ID random | learned default `g=1.0` | 1.00 | 0.6930 | +0.0135 | 155.8 | +18.7 | 3.5254 | 0.3121 | 0.002782 | 0.002014 | 48.26 |
| ID random | calibrated `g=0.5`, `goal=3.5`, `smooth=0.5` | 1.00 | 0.6733 | -0.0062 | 132.0 | -5.1 | 3.5714 | 0.3295 | 0.003493 | 0.003626 | 49.41 |
| ID random | calibrated `g=0.5`, `goal=3.5`, `smooth=1.0` | 1.00 | 0.6766 | -0.0029 | 129.8 | -7.2 | 3.5750 | 0.3282 | 0.003401 | 0.002862 | 48.97 |
| OOD obstacle | nominal CUDA | 1.00 | 0.6768 | +0.0000 | 142.3 | +0.0 | 2.7875 | 0.3269 | 0.003156 | 0.002765 | 7.19 |
| OOD obstacle | learned default `g=1.0` | 1.00 | 0.6878 | +0.0109 | 175.1 | +32.8 | 2.8150 | 0.3133 | 0.002764 | 0.002156 | 53.85 |
| OOD obstacle | calibrated `g=0.5`, `goal=3.5`, `smooth=0.5` | 1.00 | 0.6717 | -0.0051 | 133.8 | -8.6 | 2.8106 | 0.3318 | 0.003503 | 0.003551 | 49.60 |
| OOD obstacle | calibrated `g=0.5`, `goal=3.5`, `smooth=1.0` | 1.00 | 0.6775 | +0.0006 | 130.4 | -11.9 | 2.8696 | 0.3302 | 0.003411 | 0.002790 | 49.00 |
| OOD terrain | nominal CUDA | 1.00 | 0.6848 | +0.0000 | 141.8 | +0.0 | 3.5747 | 0.3633 | 0.003084 | 0.002728 | 6.43 |
| OOD terrain | learned default `g=1.0` | 1.00 | 0.6900 | +0.0051 | 153.7 | +11.8 | 3.5101 | 0.3409 | 0.002704 | 0.001871 | 51.39 |
| OOD terrain | calibrated `g=0.5`, `goal=3.5`, `smooth=0.5` | 1.00 | 0.6774 | -0.0074 | 138.4 | -3.4 | 3.5778 | 0.3680 | 0.003434 | 0.003516 | 49.73 |
| OOD terrain | calibrated `g=0.5`, `goal=3.5`, `smooth=1.0` | 1.00 | 0.6785 | -0.0063 | 130.2 | -11.7 | 3.5788 | 0.3636 | 0.003330 | 0.002775 | 49.22 |

- Conclusion:
  - The Stage 5-C hypothesis is supported: full residual correction was too strong for random tasks. Reducing residual gain to `0.5` and increasing `goal_xy_weight` to `3.5` removes the default learned controller's final-distance and steps regression on ID/OOD 20-episode runs.
  - `smooth_weight=1.0` is the better current calibrated candidate for larger benchmarking because it has the best steps across ID/OOD while keeping final distance at or slightly better than nominal. `smooth_weight=0.5` gives the best final distance in most scenarios but increases smoothness/jerk cost metrics.
  - The calibration trades away part of the default learned controller's lower terrain-risk and smoother-control behavior. Default `g=1.0` remains the conservative/smooth reference, while calibrated `g=0.5`, `goal=3.5`, `smooth=1.0` is the efficiency candidate.
  - Stage 5-D should compare at least nominal CUDA, learned default `g=1.0`, and calibrated learned `g=0.5/goal=3.5/smooth=1.0` on 50-100 episodes. If terrain-risk/smoothness advantages are required in the same controller, run a small Pareto sweep over obstacle/terrain/smoothness-related weights before the larger benchmark.

### 2026-05-02: S5-009 Pareto Sweep Around Calibrated Learned Controller

- Goal: decide whether to enter Stage 5-D directly or first look for a more balanced learned controller near the S5-008 efficiency candidate.
- Scope:
  - Full ID random grid: `27` cases x `10` episodes.
  - OOD validation: `3` selected candidates x `10` episodes on OOD obstacle and OOD terrain.
  - References: nominal CUDA and default learned `residual_gain=1.0` on ID/OOD `10` episodes.
- Seed mapping: `seed = base_seed + episode_id`, `base_seed=123`, `episode_id=0..9`.
- Backend/model:
  - backend: `cuda`
  - learned device: `cuda`
  - model dir: `results/fdm_baselines/stage4_mlp_seed123_hardened`
  - checkpoint: `best_model.pt`
  - normalization: `normalization.npz`
- Full ID grid:
  - Command: `python3 tools/sweep_stage5_calibration.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name s5_009_id_pareto_10ep --output results/stage5_calibration/s5_009/id_random_pareto10_seed123_cuda --episodes 10 --base-seed 123 --backend cuda --controllers learned --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --residual-gains 0.4,0.5,0.6 --cost-grid goal_xy_weight=3.0,3.5,4.0 --cost-grid smooth_weight=0.75,1.0,1.25`
  - Output: `results/stage5_calibration/s5_009/id_random_pareto10_seed123_cuda/stage5_calibration_sweep_summary.json`
- Reference outputs:
  - `results/stage5_calibration/s5_009/id_random_reference10_seed123_cuda/stage5_benchmark_summary.json`
  - `results/stage5_calibration/s5_009/ood_obstacle_reference10_seed123_cuda/stage5_benchmark_summary.json`
  - `results/stage5_calibration/s5_009/ood_terrain_reference10_seed123_cuda/stage5_benchmark_summary.json`
- Selected OOD validation candidates:
  - balanced: `residual_gain=0.5`, `goal_xy_weight=3.0`, `smooth_weight=0.75`
  - S5-008 current: `residual_gain=0.5`, `goal_xy_weight=3.5`, `smooth_weight=1.0`
  - aggressive efficiency: `residual_gain=0.6`, `goal_xy_weight=4.0`, `smooth_weight=1.0`
- Failure check:
  - S5-009 episode summaries checked: `390`
  - failed or unsuccessful: `0`

| Scenario | Controller | Success | Final | Delta Final | Steps | Delta Steps | Clearance | Delta Clear | Risk | Delta Risk | Smooth | Delta Smooth | Jerk | Delta Jerk | MPPI ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID random | nominal | 1.00 | 0.6774 | +0.0000 | 136.0 | +0.0 | 3.4335 | +0.0000 | 0.3163 | +0.0000 | 0.003232 | +0.000000 | 0.002868 | +0.000000 | 5.64 |
| ID random | default `g=1.0` | 1.00 | 0.6928 | +0.0154 | 168.4 | +32.4 | 3.3571 | -0.0763 | 0.2978 | -0.0185 | 0.002786 | -0.000446 | 0.002120 | -0.000748 | 51.64 |
| ID random | balanced `0.5/3.0/0.75` | 1.00 | 0.6740 | -0.0034 | 132.6 | -3.4 | 3.3765 | -0.0570 | 0.3162 | -0.0001 | 0.003307 | +0.000075 | 0.002875 | +0.000007 | 50.74 |
| ID random | current `0.5/3.5/1.0` | 1.00 | 0.6789 | +0.0015 | 129.1 | -6.9 | 3.4371 | +0.0036 | 0.3180 | +0.0017 | 0.003459 | +0.000227 | 0.002816 | -0.000052 | 49.64 |
| ID random | efficiency `0.6/4.0/1.0` | 1.00 | 0.6761 | -0.0013 | 123.9 | -12.1 | 3.4352 | +0.0017 | 0.3216 | +0.0053 | 0.003540 | +0.000309 | 0.003201 | +0.000332 | 50.12 |
| OOD obstacle | nominal | 1.00 | 0.6777 | +0.0000 | 140.5 | +0.0 | 2.9387 | +0.0000 | 0.3164 | +0.0000 | 0.003202 | +0.000000 | 0.002771 | +0.000000 | 5.17 |
| OOD obstacle | default `g=1.0` | 1.00 | 0.6862 | +0.0086 | 167.5 | +27.0 | 2.9933 | +0.0546 | 0.3078 | -0.0087 | 0.002727 | -0.000475 | 0.002062 | -0.000709 | 50.99 |
| OOD obstacle | balanced `0.5/3.0/0.75` | 1.00 | 0.6792 | +0.0015 | 133.3 | -7.2 | 3.0016 | +0.0629 | 0.3240 | +0.0076 | 0.003351 | +0.000150 | 0.002871 | +0.000100 | 52.77 |
| OOD obstacle | current `0.5/3.5/1.0` | 1.00 | 0.6765 | -0.0012 | 129.9 | -10.6 | 3.0021 | +0.0634 | 0.3200 | +0.0035 | 0.003482 | +0.000280 | 0.002775 | +0.000004 | 50.54 |
| OOD obstacle | efficiency `0.6/4.0/1.0` | 1.00 | 0.6738 | -0.0038 | 125.6 | -14.9 | 3.0298 | +0.0912 | 0.3228 | +0.0064 | 0.003560 | +0.000358 | 0.003107 | +0.000335 | 49.95 |
| OOD terrain | nominal | 1.00 | 0.6872 | +0.0000 | 141.1 | +0.0 | 3.4548 | +0.0000 | 0.3512 | +0.0000 | 0.003183 | +0.000000 | 0.002735 | +0.000000 | 5.51 |
| OOD terrain | default `g=1.0` | 1.00 | 0.6892 | +0.0020 | 167.3 | +26.2 | 3.3280 | -0.1269 | 0.3174 | -0.0338 | 0.002722 | -0.000461 | 0.001868 | -0.000867 | 52.43 |
| OOD terrain | balanced `0.5/3.0/0.75` | 1.00 | 0.6817 | -0.0056 | 135.0 | -6.1 | 3.3819 | -0.0730 | 0.3508 | -0.0004 | 0.003200 | +0.000017 | 0.002765 | +0.000030 | 49.86 |
| OOD terrain | current `0.5/3.5/1.0` | 1.00 | 0.6824 | -0.0048 | 127.4 | -13.7 | 3.4533 | -0.0016 | 0.3504 | -0.0008 | 0.003398 | +0.000215 | 0.002756 | +0.000021 | 51.00 |
| OOD terrain | efficiency `0.6/4.0/1.0` | 1.00 | 0.6726 | -0.0146 | 128.5 | -12.6 | 3.4046 | -0.0502 | 0.3546 | +0.0033 | 0.003528 | +0.000345 | 0.002981 | +0.000246 | 51.28 |

- Cross-scenario average deltas versus nominal:
  - balanced `0.5/3.0/0.75`: final `-0.0025`, steps `-5.57`, risk `+0.0023`, smooth `+0.000081`, jerk `+0.000046`
  - current `0.5/3.5/1.0`: final `-0.0015`, steps `-10.4`, risk `+0.0015`, smooth `+0.000241`, jerk `-0.000009`
  - efficiency `0.6/4.0/1.0`: final `-0.0066`, steps `-13.2`, risk `+0.0050`, smooth `+0.000337`, jerk `+0.000304`
- Conclusion:
  - The Pareto sweep found a real balanced candidate: `residual_gain=0.5`, `goal_xy_weight=3.0`, `smooth_weight=0.75`. It preserves 100% success, improves average final distance and steps, and keeps terrain risk/smoothness/jerk close to nominal.
  - It does not dominate every metric: OOD obstacle final distance is slightly worse than nominal by `+0.0015`, and clearance can be slightly lower in ID/OOD terrain. Treat it as a balanced candidate, not a final winner.
  - `residual_gain=0.6`, `goal_xy_weight=4.0`, `smooth_weight=1.0` is the aggressive efficiency candidate. It has the strongest final-distance/steps gains but the largest risk/smoothness/jerk penalty.
  - Stage 5-D should include nominal CUDA, default learned `g=1.0`, S5-008 current efficiency `0.5/3.5/1.0`, S5-009 balanced `0.5/3.0/0.75`, and optionally aggressive efficiency `0.6/4.0/1.0` if runtime budget allows.

### 2026-05-02: S5-010 Stage 5-D 50-Episode ID/OOD Benchmark

- Goal: confirm whether the S5-009 candidate ranking is stable at larger scale before making Stage 5 paper-level claims.
- Scope:
  - Scenarios: ID random, OOD obstacle, OOD terrain.
  - Official controllers: nominal CUDA, default learned `residual_gain=1.0`, current efficiency `0.5/3.5/1.0`, balanced `0.5/3.0/0.75`.
  - Episodes: `50` per official controller/scenario.
  - Seed mapping: `seed = base_seed + episode_id`, `base_seed=123`, `episode_id=0..49`.
- Backend/model:
  - backend: `cuda`
  - learned device: `cuda`
  - model dir: `results/fdm_baselines/stage4_mlp_seed123_hardened`
  - checkpoint: `best_model.pt`
  - normalization: `normalization.npz`
- Execution:
  - The matrix was run in parallel with up to `3` benchmark processes because the sequential runner was too slow.
  - Official output: `results/stage5_d/s5_010_parallel`
  - Official summary CSV: `results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.csv`
  - Official summary JSON: `results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json`
  - Aggressive efficiency `0.6/4.0/1.0` was started as an optional runtime-budget group but stopped and excluded from official S5-010 reporting.
- Failure check:
  - Official aggregate rows: `12`
  - Official aggregate errors: `0`
  - All official controller/scenario groups completed `50` episodes with `success_rate=1.0`.

| Scenario | Controller | Success | Final | Delta Final | Steps | Delta Steps | Clearance | Delta Clear | Risk | Delta Risk | Smooth | Delta Smooth | Jerk | Delta Jerk | MPPI ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID random | nominal | 1.00 | 0.6808 | +0.0000 | 132.7 | +0.0 | 3.4794 | +0.0000 | 0.3404 | +0.0000 | 0.003192 | +0.000000 | 0.002735 | +0.000000 | 6.43 |
| ID random | default `g=1.0` | 1.00 | 0.6908 | +0.0101 | 155.0 | +22.3 | 3.4873 | +0.0078 | 0.3198 | -0.0206 | 0.002795 | -0.000397 | 0.001919 | -0.000816 | 57.69 |
| ID random | current `0.5/3.5/1.0` | 1.00 | 0.6785 | -0.0022 | 125.8 | -6.9 | 3.5157 | +0.0363 | 0.3424 | +0.0021 | 0.003405 | +0.000213 | 0.002855 | +0.000120 | 60.99 |
| ID random | balanced `0.5/3.0/0.75` | 1.00 | 0.6799 | -0.0009 | 129.6 | -3.1 | 3.5078 | +0.0283 | 0.3403 | -0.0001 | 0.003316 | +0.000125 | 0.002838 | +0.000103 | 58.52 |
| OOD obstacle | nominal | 1.00 | 0.6784 | +0.0000 | 134.8 | +0.0 | 2.8470 | +0.0000 | 0.3388 | +0.0000 | 0.003191 | +0.000000 | 0.002754 | +0.000000 | 6.07 |
| OOD obstacle | default `g=1.0` | 1.00 | 0.6890 | +0.0105 | 165.6 | +30.8 | 2.8710 | +0.0240 | 0.3198 | -0.0191 | 0.002798 | -0.000393 | 0.002034 | -0.000720 | 53.89 |
| OOD obstacle | current `0.5/3.5/1.0` | 1.00 | 0.6778 | -0.0007 | 127.1 | -7.7 | 2.9019 | +0.0549 | 0.3419 | +0.0031 | 0.003425 | +0.000234 | 0.002815 | +0.000061 | 52.86 |
| OOD obstacle | balanced `0.5/3.0/0.75` | 1.00 | 0.6812 | +0.0027 | 130.7 | -4.0 | 2.9056 | +0.0586 | 0.3398 | +0.0009 | 0.003297 | +0.000105 | 0.002838 | +0.000083 | 52.36 |
| OOD terrain | nominal | 1.00 | 0.6852 | +0.0000 | 134.0 | +0.0 | 3.4964 | +0.0000 | 0.3760 | +0.0000 | 0.003124 | +0.000000 | 0.002640 | +0.000000 | 5.79 |
| OOD terrain | default `g=1.0` | 1.00 | 0.6886 | +0.0034 | 154.1 | +20.1 | 3.4684 | -0.0280 | 0.3503 | -0.0257 | 0.002688 | -0.000437 | 0.001744 | -0.000896 | 53.44 |
| OOD terrain | current `0.5/3.5/1.0` | 1.00 | 0.6785 | -0.0067 | 127.0 | -6.9 | 3.5196 | +0.0232 | 0.3783 | +0.0023 | 0.003404 | +0.000280 | 0.002781 | +0.000141 | 53.40 |
| OOD terrain | balanced `0.5/3.0/0.75` | 1.00 | 0.6835 | -0.0017 | 131.5 | -2.5 | 3.4890 | -0.0073 | 0.3750 | -0.0010 | 0.003213 | +0.000089 | 0.002754 | +0.000114 | 52.70 |

Cross-scenario learned deltas versus nominal:

| Candidate | Delta Final | Delta Steps | Delta Risk | Delta Smooth | Delta Jerk | Delta MPPI ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default `g=1.0` | +0.00798 | +24.43 | -0.02179 | -0.000409 | -0.000811 | +48.91 |
| current `0.5/3.5/1.0` | -0.00322 | -7.15 | +0.00247 | +0.000242 | +0.000107 | +49.65 |
| balanced `0.5/3.0/0.75` | +0.00005 | -3.20 | -0.00002 | +0.000106 | +0.000100 | +48.43 |

- Conclusion:
  - The 50-episode Stage 5-D result supports a three-mode framing, not a single all-metric winner.
  - Default learned `g=1.0` is the conservative/smooth mode: it keeps `100%` success and reduces terrain risk, smoothness, and jerk, but regresses final distance and steps.
  - Current efficiency `0.5/3.5/1.0` is the efficiency mode: it has the strongest official final-distance and steps improvements among the four official groups, but slightly increases risk/smoothness/jerk.
  - Balanced `0.5/3.0/0.75` is a balanced operating-point candidate: it improves steps, keeps cross-scenario final distance essentially tied with nominal, and keeps terrain risk essentially tied/slightly lower than nominal, with a small smoothness/jerk penalty.
  - Learned runtime remains much slower than nominal CUDA: learned controllers are roughly `52-61 ms` per MPPI step versus nominal `5.8-6.4 ms`. This is acceptable for offline Stage 5 benchmarking, but should not be claimed as equivalent to the nominal real-time envelope.
  - Next step: record Stage 5-D result framing for paper figures and run focused runtime profiling/optimization. Expand to `100` episodes only if the paper needs tighter confidence intervals or if candidate differences remain too small.

### 2026-05-02: S6-001 Paper-Ready Stage 5 Result Package

- Goal: organize S5-010 into paper-ready figures, tables, and a compressed result package without adding new algorithms or tuning more parameters.
- Source result:
  - `results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.json`
  - `results/stage5_d/s5_010_parallel/s5_010_official_50ep_summary.csv`
- Command:
  - `python3 tools/plot_stage5_results.py`
- Generated tracked files:
  - `tools/plot_stage5_results.py`
  - `docs/agent_memory/STAGE6_RESULT_PACKAGE.md`
  - `figures/stage5/stage5_main_result_bars.png`
  - `figures/stage5/stage5_paired_delta_boxplots.png`
  - `figures/stage5/stage5_pareto_scatter.png`
  - `figures/stage5/stage5_trajectory_gallery.png`
  - `figures/stage5/stage5_runtime_bars.png`
  - `figures/stage5/stage5_failure_tradeoff_analysis.png`
  - `tables/stage5/stage5_main_results.csv/md`
  - `tables/stage5/stage5_operating_modes.csv/md`
  - `tables/stage5/stage5_paired_delta_summary.csv/md`
  - `tables/stage5/stage5_paired_episode_deltas.csv`
  - `tables/stage5/stage5_runtime.csv/md`
  - `tables/stage5/stage5_result_package_manifest.json`
- Visualization note:
  - trajectory gallery plots the goal marker and a dashed goal-tolerance circle from `simulation.minimum_distance`, so successful trajectories do not need to terminate exactly on the star marker.
- Generated local package:
  - `results/stage6_result_package/stage5_result_package.zip`
- Style:
  - white background
  - unified font
  - fixed controller colors: nominal blue, default learned orange, efficiency green, balanced purple
- Key package metrics:
  - official aggregate rows: `12`
  - paired episode delta rows: `450`
  - zip files included: `19`
- Conclusion:
  - S6-001 packages Stage 5-D around the claim that learned-FDM-MPPI supports multiple calibrated operating modes. It preserves the boundary that no controller is an all-metric winner.
  - The most important new figure is `stage5_paired_delta_boxplots.png`, which shows per-episode learned-minus-nominal deltas for final distance, steps, clearance, terrain risk, smoothness, jerk, and runtime.
  - Next step is S6-002 runtime profiling/optimization, unless a 100-episode confirmation is needed for tighter confidence intervals.

### 2026-05-02: S6-001 Addendum - Seed123 Oracle Parameter GIFs

- Goal: run the fixed `seed=123` scene from `config/b2_omni_oracle.yaml` for the Stage 5 operating modes and save inspectable GIFs.
- Command:
  - `python3 tools/run_stage5_seed123_param_gifs.py`
  - `python3 tools/plot_stage5_results.py`
- Output:
  - `results/stage6_result_package/seed123_oracle_param_gifs/index.html`
  - `results/stage6_result_package/seed123_oracle_param_gifs/gifs/nominal_seed123_animation.gif`
  - `results/stage6_result_package/seed123_oracle_param_gifs/gifs/default_g10_seed123_animation.gif`
  - `results/stage6_result_package/seed123_oracle_param_gifs/gifs/efficiency_g05_goal35_smooth10_seed123_animation.gif`
  - `results/stage6_result_package/seed123_oracle_param_gifs/gifs/balanced_g05_goal30_smooth075_seed123_animation.gif`
  - Repacked zip: `results/stage6_result_package/stage5_result_package.zip`
- Metrics:
  - Nominal CUDA: success `true`, final distance `0.3461`, steps `225`, mean MPPI `5.26 ms`.
  - Default learned `g=1.0`: success `true`, final distance `0.3371`, steps `214`, mean MPPI `25.63 ms`.
  - Efficiency `0.5/3.5/1.0`: success `true`, final distance `0.3484`, steps `211`, mean MPPI `25.48 ms`.
  - Balanced `0.5/3.0/0.75`: success `true`, final distance `0.3282`, steps `230`, mean MPPI `25.14 ms`.
- Visualization note:
  - `animation.gif` and `trajectory.png` now draw the dashed goal-tolerance circle from `simulation.minimum_distance`, so the successful stop position is visually explained.
- Verification:
  - GIF frame counts: nominal `225`, default `214`, efficiency `211`, balanced `230`.
  - Zip package includes the seed123 oracle HTML, CSV/JSON summary, copied GIFs, and per-run artifacts.

### 2026-05-02: S5-E5 / S6 Risk-Aware Learned-FDM-MPPI Result Package

- Goal: complete a paper-ready risk-aware learned-FDM-MPPI result package using same-backend Torch comparisons, explicit terrain-risk MPPI cost, paired statistics, Nature-style figures, and a fixed two-obstacle visual benchmark.
- Baseline check:
  - Branch created from latest `origin/fdm`: `codex/s5-e5-risk-aware-result-package`.
  - PR #26 baseline was present on `fdm`: `MppiOmniTorch`, Torch learned controller reuse, `create_omni_controller(... backend=torch)`, Torch benchmark path, risk-aware analyzer, and Torch MPPI tests.
  - Initial validation: `python3 -m pytest -q` passed with `173 passed`.
- Official method:
  - Backend: `torch`.
  - Device: `cuda`.
  - Seed mapping: `seed = 123 + episode_id`.
  - Learned setting: `residual_gain=0.5`, `goal_xy_weight=3.0`, `smooth_weight=0.75`.
  - Risk cost: `terrain_risk_mode=excess`, `terrain_risk_power=2.0`, `terrain_risk_threshold=0.3`.
- Risk-weight selection:
  - Command template: `python3 tools/sweep_stage5_e_risk_cost.py --configs <map.yaml> --episodes 10 --base-seed 123 --backend torch --controllers nominal,learned --risk-weights 0,0.5,1,3,5,10 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-device cuda --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75`.
  - Selected official weights: low_friction_patch `10`, safe_corridor `0.5`, risk_band `5`, two_obstacle_standard `3`.
- Official 50-episode outputs:
  - `results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10`
  - `results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5`
  - `results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5`
  - `results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3`
  - Each output contains `stage5_e_risk_sweep_summary.json`, per-case `stage5_benchmark_summary.json`, and `analysis/` paired statistics.
- Fixed two-obstacle visual command:
  - `python3 tools/visualize_stage5_closed_loop.py --config config/b2_omni_oracle.yaml --scenario-name two_obstacle_standard --output results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123 --seed 123 --backend torch --fdm-device cuda --fdm-residual-gain 0.5 --risk-aware-2x2 --risk-weight 3 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75`
  - Output: `results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123/stage5_e_visual_eval_summary.json`.
  - Seed123 metrics: risk-aware learned final `0.3394`, steps `220`, cumulative risk `94.31`, excess `31.74`, exposure `0.7909`, mean MPPI `26.53 ms`; it has the lowest risk metrics among the four visual cases.
- Paper figure command:
  - `python3 tools/plot_stage5_e_risk_aware_results.py --sweep-summary results/stage5_e_risk_aware/s5_e5_official_50ep_low_friction_w10/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_safe_corridor_w0_5/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_risk_band_w5/stage5_e_risk_sweep_summary.json,results/stage5_e_risk_aware/s5_e5_official_50ep_two_obstacle_w3/stage5_e_risk_sweep_summary.json --visual-summary results/stage5_e_risk_aware/s5_e5_two_obstacle_visual_seed123/stage5_e_visual_eval_summary.json --output figures/stage5_e --tables-output tables/stage5_e`
  - Generated `8` case figures, `table_s5e_main_results.csv`, `table_s5e_paired_stats.csv`, and copied `fig_s5e_two_obstacle_animation.gif`.
- Key risk-on learned-minus-nominal deltas:
  - low_friction_patch, weight `10`: final `-0.0127`, cumulative risk `-9.1263`, excess risk `-3.7318`, exposure `-0.0204`, runtime `+21.57 ms`.
  - safe_corridor, weight `0.5`: final `-0.0019`, cumulative risk `-2.2375`, excess risk `+0.0040`, exposure `+0.0033`, runtime `+27.94 ms`.
  - risk_band, weight `5`: final `-0.0002`, cumulative risk `-3.3511`, excess risk `-0.3686`, exposure `+0.0005`, runtime `+21.08 ms`.
  - two_obstacle_standard, weight `3`: final `-0.0071`, cumulative risk `-7.6334`, excess risk `-2.7628`, exposure `+0.0139`, runtime `+17.65 ms`.
- Statistical boundary:
  - low_friction_patch supports the strongest risk-aware claim: cumulative risk, excess risk, and exposure have bootstrap CIs below zero and Wilcoxon p-values below `0.001`.
  - safe_corridor supports cumulative-risk reduction but not excess/exposure improvement.
  - risk_band remains a stress-test limitation: cumulative risk improves, while excess/exposure are not clean wins.
  - two_obstacle_standard is strong visual continuity evidence; final/cumulative/excess improve significantly, but exposure does not.
- Tracked package:
  - `tools/plot_stage5_e_risk_aware_results.py`
  - `tools/analyze_stage5_e_risk_aware.py`
  - `tools/visualize_stage5_closed_loop.py`
  - `docs/agent_memory/NATURE_FIGURE_STYLE.md`
  - `docs/agent_memory/STAGE5_E_RISK_AWARE_PROTOCOL.md`
  - `docs/agent_memory/STAGE5_E_RISK_AWARE_RESULTS.md`
  - `figures/stage5_e/`
  - `tables/stage5_e/`
- Boundary:
  - Do not claim learned-FDM-MPPI dominates all maps or metrics.
  - Do not label closed-loop trajectories as GT.
  - Do not use backend `cuda` PyCUDA-vs-Torch mixed runs as official risk-aware evidence.
  - Do not claim learned Torch runtime is equivalent to nominal Torch.

### 2026-05-03: S6-002 Runtime Profiling And First Terrain Sampling Optimization

- Goal: execute the next Stage 6 step after the paper-ready Stage 5 package by profiling learned Torch/CUDA runtime and applying a low-risk optimization.
- Branch:
  - `codex/s6-runtime-profiling`
  - Base: `origin/fdm @ 16b043b` after PR #27 was merged into `fdm`.
- Root-cause method:
  - Used existing `tools/profile_stage5_learned_torch.py` runtime buckets.
  - Re-ran 20-step profile on standard, ID random, and low_friction_patch configs.
  - Confirmed rollout-side terrain feature computation remains the largest learned runtime bucket, especially on noise-enabled random terrain.
- Pre-optimization commands:
  - `python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle.yaml --output results/stage6_runtime_profile/standard_seed123_g05_20steps --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5`
  - `python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_random100_dataset.yaml --output results/stage6_runtime_profile/id_random_seed123_g05_20steps --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5`
  - `python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_low_friction_patch.yaml --output results/stage6_runtime_profile/low_friction_seed123_g05_20steps --steps 20 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5`
- Key pre-optimization buckets:
  - Standard: `rollout_total_ms=992.23`, `terrain_features_ms=357.61`, `fdm_inference_ms=177.20`, `state_integrate_ms=163.97`, `obstacle_cost_ms=32.47`.
  - ID random auto-seed: `rollout_total_ms=1315.42`, `terrain_features_ms=784.27`, `fdm_inference_ms=142.68`, `state_integrate_ms=155.78`, `obstacle_cost_ms=68.05`.
  - low_friction_patch: `rollout_total_ms=1072.69`, `terrain_features_ms=445.62`, `fdm_inference_ms=162.27`, `state_integrate_ms=166.90`, `obstacle_cost_ms=6.80`.
- Change:
  - Added `MppiOmniTorch._bilinear_sample_many_torch()` and `noise_fields_t`.
  - `MppiOmniTorch._terrain_features_torch()` now samples noise, x-gradient, and y-gradient terrain fields in one pass instead of recomputing bilinear indices/weights three times.
- TDD evidence:
  - Red test first: `python3 -m pytest tests/test_mppi_omni_torch.py::test_nominal_torch_bilinear_sample_many_matches_individual_samples -q` failed with missing `_bilinear_sample_many_torch`.
  - After implementation: same test passed.
  - Targeted regression passed: `python3 -m pytest tests/test_mppi_omni_torch.py::test_nominal_torch_batch_cost_matches_numpy_with_terrain_risk tests/test_mppi_omni_learned_torch.py::test_learned_torch_rollout_scales_residual_with_gain -q`.
- Microbenchmark:
  - 1024 CPU query points, 32x32 noise grids.
  - Three individual samplers: `0.369392 ms/call`.
  - Batched sampler: `0.187556 ms/call`.
  - Sampling subroutine speedup: `1.970x`.
- Post-optimization profile:
  - ID random auto-seed `terrain_features_ms` went from `784.27` to `525.28` (`-33.02%`).
  - Standard and low_friction are within profiling variance because they do not stress the repeated noise-grid path as strongly.
  - Caveat: `config/b2_omni_oracle_random100_dataset.yaml` uses `scenario.random_seed: auto`, so this is hotspot evidence, not a paired trajectory benchmark.
- Post-merge smoke:
  - Command: `python3 tools/profile_stage5_learned_torch.py --config config/b2_omni_oracle_random100_dataset.yaml --output results/stage6_runtime_profile/post_merge_id_random_batched_noise_5steps --steps 5 --seed 123 --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-device cuda --fdm-residual-gain 0.5`
  - Output: `results/stage6_runtime_profile/post_merge_id_random_batched_noise_5steps/`.
  - Key buckets: `rollout_total_ms=273.69`, `terrain_features_ms=166.17`, `fdm_inference_ms=41.39`, `obstacle_cost_ms=35.35`.
- Output:
  - Raw profile outputs: `results/stage6_runtime_profile/` (not tracked).
  - Tracked protocol: `docs/agent_memory/STAGE6_RUNTIME_PROFILE.md`.
- Boundary:
  - This is first-pass runtime optimization, not full learned-runtime closure.
  - Learned Torch remains slower than nominal.
  - Next candidates: paired runtime profiler with fixed scenario seeds, Torch compile/horizon-loop fusion, safe terrain/risk caching where timestep semantics match, and dense-obstacle cost profiling.

### 2026-05-03: S6-002 Fixed-Seed 2x2 Runtime Matrix

- Goal: continue Stage 6 runtime work by replacing auto-seed single-controller profile evidence with a fixed-seed same-backend 2x2 runtime matrix.
- Branch:
  - `codex/s6-runtime-profiling`
- Change:
  - Added `tools/profile_stage6_runtime_matrix.py`.
  - The tool runs `nominal_risk_off`, `nominal_risk_on`, `learned_risk_off`, and `learned_risk_on` under backend `torch`.
  - Random-start-goal configs use explicit `scenario.random_seed = base_seed + episode_id`, so case comparisons are paired by seed.
  - Nominal profiling uses `mppi.profile_enabled`; learned profiling uses `fdm.profile_enabled`.
- TDD evidence:
  - Red test first: `python3 -m pytest tests/test_stage6_runtime_matrix.py -q` failed because `tools/profile_stage6_runtime_matrix.py` did not exist.
  - After implementation: `python3 -m pytest tests/test_stage6_runtime_matrix.py tests/test_stage5_runtime_profile.py -q` passed with `2 passed`.
- Real paired profiler smoke:
  - Command: `python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed --output results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps --episodes 3 --steps 5 --base-seed 123 --backend torch --device auto --risk-weight 3 --risk-power 2.0 --risk-threshold 0.3 --risk-mode excess --fdm-model-dir results/fdm_baselines/stage4_mlp_seed123_hardened --fdm-checkpoint best_model.pt --fdm-normalization normalization.npz --fdm-residual-gain 0.5 --learned-goal-xy-weight 3.0 --learned-smooth-weight 0.75`
  - Output: `results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps/stage6_runtime_matrix_summary.json`.
- Key aggregate profile means:
  - nominal_risk_off: `mean_mppi_time_ms=23.76`, `profile_mean_rollout_total_ms=10.71`, `terrain_risk_cost_ms=0.053`.
  - nominal_risk_on: `mean_mppi_time_ms=18.87`, `profile_mean_rollout_total_ms=10.28`, `terrain_risk_cost_ms=4.424`.
  - learned_risk_off: `mean_mppi_time_ms=44.55`, `profile_mean_rollout_total_ms=40.35`, `terrain_features_ms=0.928`, `fdm_inference_ms=0.232`, `terrain_risk_cost_ms=0.054`.
  - learned_risk_on: `mean_mppi_time_ms=44.09`, `profile_mean_rollout_total_ms=38.85`, `terrain_features_ms=0.913`, `fdm_inference_ms=0.192`, `terrain_risk_cost_ms=1.122`.
- Key paired deltas:
  - learned_risk_on vs nominal_risk_on: `mean_mppi_time_ms_delta=+25.22`, `profile_mean_rollout_total_ms_delta=+28.57`, `terrain_risk_cost_ms_delta=-3.302`.
  - learned_risk_off vs nominal_risk_off: `mean_mppi_time_ms_delta=+20.78`, `profile_mean_rollout_total_ms_delta=+29.64`.
  - learned_risk_on vs learned_risk_off: `mean_mppi_time_ms_delta=-0.46`, `profile_mean_rollout_total_ms_delta=-1.50`, `terrain_risk_cost_ms_delta=+1.068`.
- Boundary:
  - This is a short profiling smoke, not a paper runtime benchmark.
  - It confirms terrain-risk cost is not the dominant learned-vs-nominal runtime gap; learned rollout/feature/integration work remains the next optimization target.

### 2026-05-03: S6-002 Torch Inference Mode Runtime Cleanup

- Goal: continue Stage 6 runtime optimization with a behavior-preserving Torch hygiene change: disable autograd for the full MPPI `compute_control()` path.
- Branch:
  - `codex/s6-runtime-profiling`
- Change:
  - Wrapped `MppiOmniTorch.compute_control()` in `torch.inference_mode()`.
  - Because `LearnedFdmMppiOmniTorch` inherits the same entry point, nominal Torch and learned Torch both run candidate sampling, rollout/cost, distribution update, and CPU transfer without autograd bookkeeping.
- TDD evidence:
  - Red test first: `python3 -m pytest tests/test_mppi_omni_torch.py::test_nominal_torch_compute_control_disables_grad_tracking -q` failed with `assert [True] == [False]`.
  - After implementation: same test passed.
  - Targeted regression passed: `python3 -m pytest tests/test_mppi_omni_torch.py tests/test_mppi_omni_learned_torch.py tests/test_stage6_runtime_matrix.py -q` with `17 passed`.
- Real paired profiler smoke:
  - Command: `python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed --output results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode --episodes 3 --steps 5 --base-seed 123 --backend torch --device cuda --risk-weight 3.0`
  - Output: `results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode/stage6_runtime_matrix_summary.json`.
- Key aggregate before -> after:
  - nominal_risk_off mean MPPI: `23.76 -> 22.75 ms`; rollout: `10.71 -> 9.43 ms`.
  - nominal_risk_on mean MPPI: `18.87 -> 17.17 ms`; rollout: `10.28 -> 8.73 ms`.
  - learned_risk_off mean MPPI: `44.55 -> 40.80 ms`; rollout: `40.35 -> 36.66 ms`; terrain features: `0.928 -> 0.850 ms`; FDM inference: `0.232 -> 0.223 ms`.
  - learned_risk_on mean MPPI: `44.09 -> 40.86 ms`; rollout: `38.85 -> 35.74 ms`; terrain features: `0.913 -> 0.852 ms`; FDM inference: `0.192 -> 0.183 ms`.
- Key paired delta:
  - learned_risk_on vs nominal_risk_on mean MPPI gap: `+25.22 -> +23.69 ms`.
  - learned_risk_on vs nominal_risk_on rollout gap: `+28.57 -> +27.01 ms`.
- Boundary:
  - This is a small runtime cleanup, not a full runtime closure.
  - Learned Torch remains materially slower than nominal Torch; next target remains per-horizon learned rollout work.

### 2026-05-03: S6-002 Runtime Matrix Delta Bucket Enhancement

- Goal: improve the fixed-seed Stage 6 runtime matrix so paired deltas expose candidate sampling and distribution update overhead, not only rollout/risk/FDM buckets.
- Change:
  - Added `profile_mean_sample_candidates_ms` and `profile_mean_update_distribution_ms` to `DELTA_METRICS` in `tools/profile_stage6_runtime_matrix.py`.
  - Extended `tests/test_stage6_runtime_matrix.py` to assert the paired delta aggregate includes both fields.
- TDD evidence:
  - Red test first: `python3 -m pytest tests/test_stage6_runtime_matrix.py -q` failed with `KeyError: 'profile_mean_sample_candidates_ms_delta_mean'`.
  - After implementation: same test passed with `1 passed`.
- Real profiler smoke:
  - Command: `python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed_delta_buckets --output results/stage6_runtime_profile/paired_id_random_fixed_seed_1ep_2steps_delta_buckets --episodes 1 --steps 2 --base-seed 123 --backend torch --device cuda --risk-weight 3.0`
  - Output: `results/stage6_runtime_profile/paired_id_random_fixed_seed_1ep_2steps_delta_buckets/stage6_runtime_matrix_summary.json`.
  - Confirmed JSON fields: `profile_mean_sample_candidates_ms_delta_mean` and `profile_mean_update_distribution_ms_delta_mean`.
- Boundary:
  - This is profiler instrumentation only; it does not change controller behavior or runtime.
  - The 1ep x 2step output is a schema smoke, not a performance conclusion.

### 2026-05-03: S6-002 Runtime Closeout Figures And 10ep Confirmation

- Goal: quickly close Stage 6 runtime profiling into a reviewable package with figures, tables, and a slightly longer fixed-seed runtime confirmation.
- New tool:
  - `tools/plot_stage6_runtime_results.py`
  - Test: `tests/test_stage6_runtime_plot.py`
- TDD evidence:
  - Red test first: `python3 -m pytest tests/test_stage6_runtime_plot.py -q` failed because `tools/plot_stage6_runtime_results.py` did not exist.
  - After implementation: same test passed.
- Closeout profiler command:
  - `python3 tools/profile_stage6_runtime_matrix.py --config config/b2_omni_oracle_random100_dataset.yaml --scenario-name id_random_fixed_seed_closeout --output results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout --episodes 10 --steps 10 --base-seed 123 --backend torch --device cuda --risk-weight 3.0`
- Closeout profiler output:
  - `results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout/stage6_runtime_matrix_summary.json`
- Runtime semantics:
  - Stage 6 profiler now sets `simulation.disable_goal_termination=true` by default.
  - `--steps 10` means forced 10 `compute_control()` calls per run, not ordinary benchmark max steps.
  - `profile_call_consistency.consistent=true`; `unique_total_calls=[10]`.
- Plot command:
  - `python3 tools/plot_stage6_runtime_results.py --summary results/stage6_runtime_profile/paired_id_random_fixed_seed_3ep_5steps_inference_mode/stage6_runtime_matrix_summary.json results/stage6_runtime_profile/paired_id_random_fixed_seed_10ep_10steps_closeout/stage6_runtime_matrix_summary.json --output figures/stage6 --tables-output tables/stage6`
- Generated figures:
  - `figures/stage6/fig_stage6_runtime_summary.png`
  - `figures/stage6/fig_stage6_runtime_summary.pdf`
  - `figures/stage6/fig_stage6_runtime_breakdown.png`
  - `figures/stage6/fig_stage6_runtime_breakdown.pdf`
  - `figures/stage6/fig_stage6_runtime_delta_buckets.png`
  - `figures/stage6/fig_stage6_runtime_delta_buckets.pdf`
- Generated tables:
  - `tables/stage6/table_stage6_runtime_summary.csv`
  - `tables/stage6/table_stage6_runtime_paired_deltas.csv`
- Closeout aggregate means:
  - nominal_risk_off: mean MPPI `13.90 ms`, rollout `8.67 ms`, profile calls/run `10`.
  - nominal_risk_on: mean MPPI `13.91 ms`, rollout `8.54 ms`, profile calls/run `10`.
  - learned_risk_off: mean MPPI `38.97 ms`, rollout `34.94 ms`, terrain features `0.834 ms`, FDM inference `0.178 ms`, profile calls/run `10`.
  - learned_risk_on: mean MPPI `39.93 ms`, rollout `34.98 ms`, terrain features `0.839 ms`, FDM inference `0.172 ms`, profile calls/run `10`.
- Closeout paired delta:
  - learned_risk_on vs nominal_risk_on: mean MPPI `+26.02 ms`, rollout `+26.44 ms`, sample `+0.003 ms`, update `+0.001 ms`, terrain-risk cost `-0.501 ms`.
- Validation:
  - `python3 -m pytest tests/test_stage6_runtime_plot.py tests/test_stage6_runtime_matrix.py tests/test_mppi_omni_torch.py tests/test_mppi_omni_learned_torch.py -q` -> `18 passed`.
  - `git diff --check` -> pass.
- Boundary:
  - Stage 6 now has a reviewable runtime profiling and plotting package.
  - It does not claim learned Torch is real-time equivalent to nominal Torch.
  - The remaining runtime limitation is explicit: learned-vs-nominal overhead is still dominated by learned rollout.

### 2026-05-03: S6-003 Final Convergence And Deployment Readiness

- Goal: close the current risk-aware learned-FDM-MPPI numerical simulation package as a reportable, reproducible, boundary-clear final package.
- Branch:
  - `codex/s6-003-final-readiness`
  - base: `origin/fdm` after PR #28 merge (`67f3371`)
- Scope boundaries:
  - No new model structure.
  - No new maps.
  - No MuJoCo or real-robot closed loop.
  - No claim that learned runtime equals nominal runtime.
  - No claim that the current stack is directly deployable on hardware.
- Added docs:
  - `docs/agent_memory/PROJECT_FINAL_STATUS.md`
  - `docs/agent_memory/STAGE6_RUNTIME_CLOSURE.md`
  - `docs/agent_memory/REAL_ROBOT_READINESS.md`
  - `docs/agent_memory/REPRODUCIBILITY_COMMANDS.md`
- Updated docs:
  - `README.md`
  - `docs/agent_memory/TASK_BOARD.md`
  - `docs/agent_memory/EXPERIMENT_LOG.md`
- Final numerical framing:
  - Strong risk-aware claim remains `low_friction_patch`.
  - `safe_corridor` is supporting evidence.
  - `risk_band` is a stress-test limitation.
  - `two_obstacle_standard` is visual continuity plus fixed-scene evidence.
  - Stage 6 runtime closure remains a limitation: forced-step closeout learned risk-on is about `+26.02 ms` mean MPPI over nominal risk-on, dominated by rollout.
- Real-robot readiness:
  - Current package is numerical simulation only.
  - Next stage may start only as read-only shadow mode.
  - Shadow-mode minimum entry conditions are listed in `REAL_ROBOT_READINESS.md`.
