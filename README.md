# b2_fdm_mppi

这是一个 ROS 2 Humble `ament_python` 包，也是 B2 全向 MPPI、oracle residual 仿真和 FDM 数据集准备的本地研发工作区。

项目 Notion 驾驶舱：

```text
https://www.notion.so/fdm_mppi-3532fb2e849f8188a4fef5bb3ae54264?t=3532fb2e849f8080ac1400a9065381ce
```

每完成一个小 stage，需要把相关实验数据、输出路径、指标、代码/配置改动、验证命令和结论更新到 Notion 对应数据库中；如果该 stage 改变了项目可见进展，还要同步更新 Notion 首页里的当前阶段、本阶段焦点 checklist、路线图状态、稳定结论或下一步。若暂时无法访问 Notion，先更新 `docs/agent_memory/` 并留下待同步说明。

当前开发线不改动 MPPI 核心控制逻辑，而是在其外围建立可复现的数据链路：

```text
Nominal B2 omni MPPI
  -> Oracle residual world
  -> Oracle episode npz collection
  -> Oracle dataset generation
  -> Residual FDM training baseline
```

## 当前阶段

当前阶段：

```text
Stage 4: residual FDM training baseline
```

Stage 3.5 已完成：

- `tools/generate_oracle_episodes.py` 支持 `--num-workers` 并行采集；
- 每个 episode 使用独立 seed：`seed = base_seed + episode_id`；
- 每个 episode 写入独立 raw result 目录：`raw_results/episode_XXXXXX`；
- manifest 按 `episode_id` 排序，episode path 使用相对路径；
- 单个 episode 失败会记录 `error` 并继续采集；
- `trajectory.csv` 保存最终 state，episode npz 不再丢最后一个 transition；
- `build_oracle_dataset.py` 和 `validate_oracle_dataset.py` 继续复用现有字段和文件格式。

Stage 4 当前目标：

- 训练一个最小 residual velocity FDM baseline；
- 输入：`states + cmd_controls + terrain_features + terrain_risk`；
- 目标：`exec_residuals = real_controls - cmd_controls`；
- 输出：`model.pt`、`normalization.npz`、`metrics.json`；
- 暂不接入 MPPI rollout。

Stage 4 不做：

- 不把 learned FDM 接入 MPPI；
- 不训练风险模型；
- 不修改 MPPI 核心控制逻辑。

下一阶段计划：

```text
Stage 5: learned FDM-MPPI integration
```

## 构建

在 ROS 2 Humble 工作空间中：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select b2_fdm_mppi
source install/setup.bash
```

仿真和训练使用的 Python 依赖记录在 `requirements.txt`。主要包括 `numpy`、`scipy`、`pandas`、`matplotlib`、`seaborn`、`PyYAML`、`jinja2`、`casadi`、`torch` 和 `pytest`。

真实 CUDA MPPI controller 需要 NVIDIA CUDA/PyCUDA 运行环境。单元测试通过 fake controller 和 CUDA skip helper 保留非 GPU 环境覆盖。

## 关键配置

```text
config/b2_omni_oracle.yaml
```

Stage 2 固定 oracle 单场景 smoke test。它是主要回归测试之一，可以生成 PNG/GIF 输出。

```text
config/b2_omni_oracle_random100.yaml
```

100x100 oracle random obstacle world，用于可视化调试。默认保留 plots，关闭 animation。

```text
config/b2_omni_oracle_random100_dataset.yaml
```

面向数据采集的 fixed-map random-task oracle 配置，用于 Stage 3.2 episode generation：

- 起点/终点随机；
- 障碍物 seed 固定；
- terrain noise seed 固定；
- 地图模式是 `fixed_map_random_tasks`；
- `results.enable_plots: false`；
- `results.enable_animation: false`。

当前 dataset 障碍设置：

```yaml
robot:
  safety_dist: 0.25

obstacles:
  random_seed: 123
  num_random: 120
```

## 常用命令

### 运行完整测试

```bash
python3 -m pytest -q
```

### Oracle Smoke Test

Stage 3 相关改动不应该破坏这个命令：

```bash
python3 tools/run_omni_mppi.py \
  --config config/b2_omni_oracle.yaml \
  --seed 123
```

期望输出包含：

```text
reached_goal=True
failed=False
```

### 100x100 可视化调试

```bash
python3 tools/run_omni_mppi.py \
  --config config/b2_omni_oracle_random100.yaml \
  --seed 123
```

这个配置默认生成：

```text
trajectory.png
oracle_diagnostics.png
```

默认不生成：

```text
animation.gif
```

如果需要调试 GIF，可以手动在 YAML 中打开 `results.enable_animation`。

### 单 Episode 采集

```bash
python3 tools/collect_oracle_episode.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --episode-id 0 \
  --seed 123 \
  --output datasets/oracle_debug/episodes/episode_000000.npz
```

这个命令中：

- `--episode-id` 写入 `episode_ids` 数组；
- `--seed` 设置本次 `scenario.random_seed` 和 `oracle_residual.seed`；
- `obstacles.random_seed` 仍由配置固定；
- `terrain.noise_seed` 仍由配置固定；
- 输出是一个 `.npz` episode 文件；
- 不生成 PNG/GIF。

### 串行多 Episode 采集

```bash
python3 tools/generate_oracle_episodes.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --episodes 20 \
  --base-seed 123 \
  --output datasets/oracle_debug
```

seed 映射：

```text
seed = base_seed + episode_id
```

输出结构：

```text
datasets/oracle_debug/
  manifest.jsonl
  summary.json
  raw_results/
    episode_000000/
    episode_000001/
    ...
  episodes/
    episode_000000.npz
    episode_000001.npz
    ...
```

### 并行多 Episode 采集

Stage 3.5 支持并行采集：

```bash
python3 tools/generate_oracle_episodes.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --episodes 500 \
  --base-seed 123 \
  --output datasets/oracle_stage3 \
  --backend numpy \
  --num-workers 4
```

建议先用 `--backend numpy` 建立可复现基线。CUDA backend 可用，但多个 worker 会同时创建 CUDA/PyCUDA 上下文，吞吐和显存风险需要单独评估。

`manifest.jsonl` 中的 `path` 是相对采集目录的路径，例如：

```text
episodes/episode_000000.npz
```

无 CUDA 环境调试时可用 NumPy backend：

```bash
python3 tools/generate_oracle_episodes.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --episodes 2 \
  --base-seed 123 \
  --output datasets/oracle_debug_numpy \
  --backend numpy
```

### 检查 Episode NPZ

```bash
python3 - <<'PY'
import numpy as np

p = "datasets/oracle_debug/episodes/episode_000000.npz"
d = np.load(p)
print(d.files)
for k in d.files:
    print(k, d[k].shape, d[k].dtype)
print("exec_residuals_match", np.allclose(
    d["exec_residuals"],
    d["real_controls"] - d["cmd_controls"],
))
PY
```

### 构建 Train / Val / Test Split

先用 Stage 3.2 生成 episode：

```bash
python3 tools/generate_oracle_episodes.py \
  --config config/b2_omni_oracle_random100_dataset.yaml \
  --episodes 20 \
  --base-seed 123 \
  --output datasets/oracle_debug
```

再合并并切分：

```bash
python3 tools/build_oracle_dataset.py \
  --input datasets/oracle_debug \
  --output datasets/oracle_debug_splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 123
```

输出结构：

```text
datasets/oracle_debug_splits/
  train.npz
  val.npz
  test.npz
  split_manifest.json
  dataset_summary.json
```

切分发生在 episode 级别，同一个 episode 的 transitions 不会跨 split。

### 验证 Dataset 质量并生成可视化

```bash
python3 tools/validate_oracle_dataset.py \
  --dataset datasets/oracle_debug_splits \
  --output datasets/oracle_debug_splits
```

输出：

```text
datasets/oracle_debug_splits/
  dataset_quality.json
  dataset_summary.png
```

`dataset_quality.json` 会记录：

```text
split_shapes
num_transitions
num_episodes
nan_count
inf_count
max_exec_residual_error
mean_exec_residual_error
zero_residual_baseline_mse
episode_leakage_check
pass
```

`dataset_summary.png` 包含 x-y 空间覆盖、control 分布、residual 分布、terrain risk、roughness/friction、episode length 和 split transition 数量。

主要 shape 预期：

```text
states (T, 6) float32
next_states (T, 6) float32
cmd_controls (T, 3) float32
real_controls (T, 3) float32
exec_residuals (T, 3) float32
oracle_residuals (T, 3) float32
terrain_features (T, 4) float32
terrain_risk (T,) float32
episode_ids (T,) int64
steps (T,) int64
```

transition 数量：

```text
T = len(residuals.csv)
```

`trajectory.csv` 会额外保存最终 state，因此 `next_states[t]` 来自第 `t + 1` 行 trajectory。

## Episode NPZ 字段

每个 `episode_XXXXXX.npz` 包含：

```text
states
next_states
cmd_controls
real_controls
exec_residuals
oracle_residuals
terrain_features
terrain_risk
episode_ids
steps
success
failed
start_goal_distance
final_distance
min_obstacle_clearance
```

字段来源：

- `states[t]`：来自 `trajectory.csv` 第 `t` 行，`[x, y, theta, vx, vy, wz]`
- `next_states[t]`：来自 `trajectory.csv` 第 `t + 1` 行
- `cmd_controls[t]`：来自 `residuals.csv`，`[cmd_vx, cmd_vy, cmd_wz]`
- `real_controls[t]`：来自 `residuals.csv`，`[real_vx, real_vy, real_wz]`
- `exec_residuals[t]`：`real_controls[t] - cmd_controls[t]`
- `oracle_residuals[t]`：来自 `residuals.csv`，`[oracle_du_vx, oracle_du_vy, oracle_du_wz]`
- `terrain_features[t]`：来自 `terrain.csv`，`[slope_f, slope_l, roughness, friction]`
- `terrain_risk[t]`：来自 `terrain.csv` 的 `risk_cost`

## Manifest 和 Summary

`manifest.jsonl` 每个 episode 一行，至少包含：

```text
episode_id
seed
path
success
failed
num_transitions
start_goal_distance
final_distance
min_obstacle_clearance
error
```

`error` 只在该 episode 采集失败时出现。单个 episode 失败不会中断整体串行采集。
单个 episode 失败也不会中断并行采集。

`summary.json` 至少包含：

```text
total_episodes
success_episodes
failed_episodes
total_transitions
success_rate
output_dir
config_path
base_seed
```

## 结果和生成数据

仿真结果写入：

```text
results/sim_results/
```

单 episode 采集会把 `.npz` 写到 `--output` 指定的位置，例如：

```text
datasets/oracle_debug/episodes/
```

除非任务明确要求，不要提交生成的 result 目录或 debug dataset。

## Stage 4 Residual FDM Baseline

训练最小 residual velocity FDM：

```bash
python3 tools/train_residual_fdm.py \
  --dataset datasets/oracle_stage3_splits \
  --output results/fdm_baselines/oracle_stage3_baseline \
  --epochs 50 \
  --batch-size 512 \
  --hidden-dim 64 \
  --learning-rate 0.001 \
  --seed 123 \
  --tensorboard-log-dir results/fdm_baselines/oracle_stage3_baseline/tensorboard
```

输入特征：

```text
states:           6
cmd_controls:     3
terrain_features: 4
terrain_risk:     1
total:           14
```

训练目标：

```text
exec_residuals: [exec_du_vx, exec_du_vy, exec_du_wz]
```

输出：

```text
model.pt
normalization.npz
metrics.json
tensorboard/events.out.tfevents.*
```

`metrics.json` 包含：

- `train_loss` / `val_loss` / `test_loss`: 标准化目标空间的 MSE；
- `val_mse` / `test_mse`: 原始 residual 单位的 MSE；
- `zero_residual_val_mse` / `zero_residual_test_mse`: 直接预测零 residual 的 baseline MSE。
- `tensorboard_log_dir`: TensorBoard event log 目录。

用 `val_mse` / `test_mse` 对比 zero-residual baseline，判断 learned residual 是否真正优于零 residual。

查看训练曲线和 residual 诊断图：

```bash
tensorboard --logdir results/fdm_baselines/oracle_stage4_baseline/tensorboard --port 6006
```

TensorBoard 记录：

- per-epoch `loss/train_standardized`、`loss/val_standardized`、`lr`；
- final `mse/val_raw`、`mse/test_raw`、zero-residual baseline MSE；
- val split residual prediction-vs-target scatter 和 residual error histogram。

### Open-loop Rollout 效果验证

训练后可以用同一个 oracle 仿真环境做 open-loop replay，对比 nominal replay、learned FDM replay 和 oracle ground truth：

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

验证流程：

```text
run oracle MPPI episode
  -> collect oracle states, cmd_controls, exec residuals
  -> replay nominal model with same cmd_controls
  -> replay learned FDM with same cmd_controls
  -> compare replay trajectories and residual predictions against oracle
```

输出：

```text
rollout_metrics.json
rollout_metrics.yaml
rollout_replay.npz
rollout_compare.gif
trajectory_compare.png
residual_compare.png
oracle_run/
```

核心指标：

- `nominal_ade_xy` / `learned_ade_xy`: replay 轨迹相对 oracle 的平均 XY 误差。
- `nominal_fde_xy` / `learned_fde_xy`: replay 终点相对 oracle 的 XY 误差。
- `horizon_metrics`: `1s`、`2s`、`4s` 窗口内的 ADE/FDE 及改善比例；同时展开为 `nominal_ade_xy_at_1s` 等顶层字段，便于脚本读取。
- `learned_vs_nominal_ade_improvement_pct`: learned replay 相对 nominal replay 的 ADE 改善比例。
- `residual_mse` / `zero_residual_mse`: learned residual 相对 oracle residual 的 MSE，以及零 residual baseline MSE。
- `residual_mse_axis`: `[vx, vy, wz]` 三轴 residual MSE。
- `command`、`git_sha`、`git_branch`、`git_dirty`、`device`、`checkpoint_path`、`normalization_path`、`gif_parameters`: 复现实验所需的运行元数据。

`rollout_compare.gif` 展示 terrain risk 热力背景、障碍物 safety boundary、oracle / nominal replay / learned FDM replay 三条轨迹同步推进，以及每帧 nominal 和 learned 相对 oracle 的当前 XY error。若只需要 JSON/PNG，可以加 `--no-gif` 跳过 GIF 渲染；此时 metrics 不会复用 output dir 中可能存在的旧 `rollout_compare.gif` 路径。

参数一致性检查：

- eval 会把实际使用的 `robot.radius`、`robot.safety_dist`、障碍物半径和可视化 safety boundary 半径写入 `rollout_metrics.json` 的 `parameter_snapshot`。
- GIF 中红色 safety boundary 使用 `obstacle_radius + robot.radius + robot.safety_dist`。
- 当前标准 eval 场景 `config/b2_omni_oracle.yaml` 使用 `robot.safety_dist: 0.25`，与训练 dataset 配置 `config/b2_omni_oracle_random100_dataset.yaml` 的障碍物安全距离一致。

`config/b2_omni_oracle.yaml` seed123 当前参考结果：

```text
oracle_reached_goal: true
oracle_steps: 219
robot_safety_dist: 0.25
visualized_safety_boundary_radii: [1.25, 1.25]
nominal_ade_xy: 0.52125
learned_ade_xy: 0.04871
nominal_fde_xy: 0.91905
learned_fde_xy: 0.14396
learned_vs_nominal_ade_improvement_pct: 90.66
nominal_ade_xy_at_1s / learned_ade_xy_at_1s: 0.01937 / 0.00108
nominal_fde_xy_at_1s / learned_fde_xy_at_1s: 0.03956 / 0.00110
nominal_ade_xy_at_2s / learned_ade_xy_at_2s: 0.04252 / 0.00101
nominal_fde_xy_at_2s / learned_fde_xy_at_2s: 0.09267 / 0.00125
nominal_ade_xy_at_4s / learned_ade_xy_at_4s: 0.09690 / 0.00154
nominal_fde_xy_at_4s / learned_fde_xy_at_4s: 0.21386 / 0.00276
residual_mse_improvement_pct: 98.94
checkpoint_path: results/fdm_baselines/stage4_mlp_seed123/model.pt
normalization_path: results/fdm_baselines/stage4_mlp_seed123/normalization.npz
rollout_compare.gif: 120 frames, 700x700
```

这个验证仍属于 Stage 4 open-loop 检查；它证明 learned FDM 能在固定控制序列 replay 中贴近 oracle，不等价于 Stage 5 的闭环 Learned-FDM-MPPI 集成。
