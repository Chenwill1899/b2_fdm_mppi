# b2_fdm_mppi

这是一个 ROS 2 Humble `ament_python` 包，也是 B2 全向 MPPI、oracle residual 仿真和 FDM 数据集准备的本地研发工作区。

当前开发线不改动 MPPI 核心控制逻辑，而是在其外围建立可复现的数据链路：

```text
Nominal B2 omni MPPI
  -> Oracle residual world
  -> Oracle episode npz collection
  -> Future FDM dataset generation
  -> Future FDM training
```

## 当前阶段

当前阶段：

```text
Stage 3.4: 数据集质量检查与数据分布可视化
```

Stage 3.4 的目标很窄：

- 读取 Stage 3.3 生成的 `train.npz` / `val.npz` / `test.npz`；
- 检查字段、shape、dtype、NaN/Inf；
- 检查 `exec_residuals = real_controls - cmd_controls`；
- 检查 train/val/test episode 无泄漏；
- 生成 `dataset_quality.json` 和 `dataset_summary.png`。

Stage 3.4 不做：

- 不训练 FDM；
- 不实现 FDM 网络；
- 不做并行采集；
- 不修改 MPPI 核心控制逻辑。

下一阶段计划：

```text
Stage 3.5: 并行采集加速
```

## 构建

在 ROS 2 Humble 工作空间中：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select b2_fdm_mppi
source install/setup.bash
```

仿真使用的 Python 依赖包括 `numpy`、`pandas`、`matplotlib`、`seaborn`、`PyYAML`、`jinja2`、`cvxpy`、`casadi` 和 `pycuda`。

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
  episodes/
    episode_000000.npz
    episode_000001.npz
    ...
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
T = len(trajectory.csv) - 1
```

最后一行 trajectory 没有 next state，因此会被丢弃。

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
