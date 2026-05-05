# FDM-MPPI 仿真工作台

这个仓库现在收敛为一条主路径，同时提供一个 profile 驱动的仿真入口：

```text
定义 experiment profile -> 选择 MPPI/controller -> 运行仿真 -> 查看轨迹/GIF/诊断图/summary
生成 oracle dataset -> 构建/校验数据集 -> 训练 residual FDM -> 评估/benchmark -> 汇总报告
```

推荐入口只有一个：

```bash
/usr/bin/python3 tools/fdm_mppi.py --help
```

当前 shell 里的 `python3` 可能被 conda 环境覆盖；需要使用系统 Python 时请直接写 `/usr/bin/python3`。

旧的 `tools/*.py` 命令保留为短期兼容 wrapper；Stage 5 / Stage 6 的实验脚本、图表和表格已放到 `archive/`，不再作为默认工作流入口。

## 目录

- `b2_fdm_mppi/controllers/`: MPPI 控制器实现。
- `b2_fdm_mppi/simulation/`: 仿真 runner、随机场景和结果写出。
- `b2_fdm_mppi/data/`: oracle episode、dataset split、dataset validator、sequence FDM V2 collector。
- `b2_fdm_mppi/training/`: residual FDM 训练、sequence FDM V2 课程训练。
- `b2_fdm_mppi/evaluation/`: dataset/rollout 评估和 closed-loop benchmark。
- `b2_fdm_mppi/experiment.py`: profile 驱动的仿真实验入口。
- `b2_fdm_mppi/reporting/`: 最小报告汇总。
- `configs/`: 当前推荐配置。
- `archive/`: 历史 stage 脚本、图表、表格和 legacy 配置快照。

## 推荐配置

- `configs/smoke.yaml`: 固定小场景回归，默认 `cuda` backend，会生成 `animation.gif`。
- `configs/dataset.yaml`: 小规模 oracle dataset 采集，默认关闭 plot/GIF 副产物。
- `configs/benchmark.yaml`: learned-FDM closed-loop benchmark 基础配置。
- `configs/experiment.yaml`: 推荐的仿真工作台 profile；在一份 YAML 里定义场景、controller、learned model 和可视化。
- `configs/mujoco_test_obstacles_localmap.yaml`: **MuJoCo + Geomapping local_costmap** 闭环配置。

旧 `config/*.yaml` 仍可用于复现实验和兼容测试，但主流程不再依赖这些路径。

## Sequence FDM V2 数据采集与训练

Sequence FDM V2 是一个端到端前向动力学模型：给定当前状态、H 步控制和地形网格，预测未来轨迹和二值风险。

### 多轨迹数据采集

同一地形上跑 K 条不同 start/goal 的轨迹，地形只生成一次：

```bash
python tools/collect_sequence_fdm_v2_data.py \
  --base-config configs/smoke.yaml \
  --episodes 300 \
  --num-trajectories 3 \
  --output-dir data/my_run \
  --base-seed 0 \
  --workers 16 \
  --map-bounds -15 15 -15 15
```

起点/终点筛选：要求 `terrain.risk_cost <= 0.5`（安全区域），失败轨迹也保留作为负样本。

每条轨迹保存为 `episode_XXXXXX_traj_YY.npz`，字段包含 `states`、`cmd_controls`、`binary_risk`、`terrain_seed`、`traj_idx`、`start_xy`、`goal_xy`。

### 课程训练

```bash
python tools/train_sequence_fdm_v2.py \
  --data-dir data/my_run \
  --output-dir checkpoints/my_run \
  --horizons 5 10 20 \
  --epochs 50 50 100 \
  --lrs 1e-3 5e-4 1e-4 \
  --batch-size 16 \
  --device cuda
```

支持 `--resume` 从上一个 checkpoint 继续训练（加载 model + optimizer 状态）。

### 3 轮迭代训练

自动编排：采集 → 训练 → resume → 重复：

```bash
python tools/run_iterative_training.py \
  --base-config configs/smoke.yaml \
  --episodes 100 100 200 \
  --num-trajectories 3 \
  --data-root data/iterative \
  --checkpoint-root checkpoints/iterative \
  --map-bounds -15 15 -15 15 \
  --device cuda
```

- Round 1: 100 eps × 3 = 300 轨迹，无 resume
- Round 2: 100 eps × 3 = 300 轨迹，resume Round 1
- Round 3: 200 eps × 3 = 600 轨迹，resume Round 2

每轮 checkpoint 保存到 `checkpoints/iterative/round_{N}/`，包含 `best_model.pt`、`optimizer.pt`、`normalization.npz`、`training_metrics.json`。



运行推荐 experiment profile：

```bash
/usr/bin/python3 tools/fdm_mppi.py experiment \
  --profile configs/experiment.yaml
```

切换 MPPI/controller：

```bash
/usr/bin/python3 tools/fdm_mppi.py experiment \
  --profile configs/experiment.yaml \
  --controller nominal_cuda

/usr/bin/python3 tools/fdm_mppi.py experiment \
  --profile configs/experiment.yaml \
  --controller nominal_numpy
```

调用学习好的 residual FDM 模型：

```bash
/usr/bin/python3 tools/fdm_mppi.py experiment \
  --profile configs/experiment.yaml \
  --controller learned_torch \
  --model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --checkpoint best_model.pt \
  --normalization normalization.npz \
  --device cuda
```

临时覆盖 backend、输出目录或可视化：

```bash
/usr/bin/python3 tools/fdm_mppi.py experiment \
  --profile configs/experiment.yaml \
  --controller nominal_cuda \
  --backend numpy \
  --output results/experiments/debug \
  --no-animation
```

`--output` 是输出根目录，最终目录仍会追加 profile 里的 `run_name`。
如果要指定本次实验的最终结果目录，使用 `--results-dir`：

```bash
/usr/bin/python3 tools/fdm_mppi.py experiment \
  --profile configs/experiment.yaml \
  --controller nominal_cuda \
  --results-dir results/experiments/manual_nominal_cuda \
  --no-animation
```

每次 experiment 会写出 `experiment_summary.json`，里面列出 `summary.json`、`trajectory.csv`、`trajectory.png`、`oracle_diagnostics.png`、`animation.gif` 等可检查产物路径。

## Experiment Profile 配置

`configs/experiment.yaml` 是推荐入口配置。它不是完整底层配置，而是 overlay：

```text
先加载 experiment.base_config -> 再应用 scenario/controllers/visualization -> 生成完整仿真配置
```

常用改动位置：

- `experiment`: 设置实验名、默认 seed、输出根目录、run name 模板，或用 `results_dir` 直接指定最终结果目录。
- `scenario`: 设置环境，包括 `initial_state`、`goal`、`world_mode`、`max_steps`、障碍物和地形。
- `scenario.terrain.goal_relief`: 终点附近的地形风险递减；`center: auto` 会跟随 `scenario.goal[:2]`，避免风险代价阻止收敛到终点。
- `controllers`: 定义可选 MPPI 方法，例如 `nominal_cuda`、`nominal_numpy`、`learned_torch`。
- `learned_fdm`: 给 learned controller 指定 `model_dir`、`checkpoint`、`normalization`、`device` 和 `residual_gain`。
- `visualization`: 控制 `trajectory.png`、`oracle_diagnostics.png` 和 `animation.gif`。

静态障碍物写在 `scenario.obstacles.virtual`：

```yaml
scenario:
  obstacles:
    static_enabled: true
    virtual:
      - [0.55, 0.18, 0.14, 0.0, 0.0, 0.0, 0.0]
      - [0.85, -0.22, 0.16, 0.0, 0.0, 0.0, 0.0]
```

每个障碍物使用 7 个字段：

```text
[x, y, radius, unused, theta, vx, vy]
```

静态场景主要使用 `x, y, radius`；后四个字段保留给动态障碍和 legacy schema 兼容。

随机障碍物可以这样启用：

```yaml
scenario:
  obstacles:
    static_enabled: true
    virtual: []
    random_enabled: true
    random_seed: 123
    num_random: 4
    radius_range: [0.12, 0.22]
    x_range: [0.25, 1.0]
    y_range: [-0.45, 0.45]
    min_obstacle_gap: 0.35
    min_start_goal_clearance: 0.35
```

切换 learned FDM 模型时，优先用命令行覆盖，不必改 profile：

```bash
/usr/bin/python3 tools/fdm_mppi.py experiment \
  --profile configs/experiment.yaml \
  --controller learned_torch \
  --model-dir results/fdm_baselines/stage4_mlp_seed123_hardened \
  --checkpoint best_model.pt \
  --normalization normalization.npz \
  --device cuda
```

## MuJoCo 闭环仿真

支持通过 ROS2 与 ausim2 MuJoCo 仿真器进行闭环控制，提供两种避障方案：

### 方案 A：虚拟障碍物（纯 MPPI）

```bash
/usr/bin/python3 tools/fdm_mppi.py mujoco-closed-loop \
  --profile configs/mujoco_test_obstacles.yaml \
  --controller nominal_numpy
```

### 方案 B：Local Costmap（Geomapping + MPPI）

需要先启动 ausim2 和 Geomapping：

```bash
# Terminal 1: 启动 ausim2
cd /home/mexxiie/prj/ausim2
./em_run.sh --headless

# Terminal 2: 启动 Geomapping
cd /home/mexxiie/prj/Geomapping_ros2
source install/setup.bash
ros2 launch traversability_mapping ausim_cube_mppi.launch.py launch_rviz:=false use_medirl:=true

# Terminal 3: 启动 MPPI
cd /home/mexxiie/prj/py-mppi
export LD_LIBRARY_PATH="/home/mexxiie/prj/Geomapping_ros2/install/elevation_msgs/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/mexxiie/prj/Geomapping_ros2/install/elevation_msgs/local/lib/python3.10/dist-packages:$PYTHONPATH"
/usr/bin/python3 tools/fdm_mppi.py mujoco-closed-loop \
  --profile configs/mujoco_test_obstacles_localmap.yaml \
  --controller nominal_numpy
```

### 一键启动脚本

```bash
./launch_mppi_sim.sh
```

该脚本自动启动 ausim2 + Geomapping + MPPI，按 `Ctrl+C` 自动清理所有进程。

低层仿真命令仍然保留：

```bash
/usr/bin/python3 tools/fdm_mppi.py run --config configs/smoke.yaml --seed 123
```

如果当前 Python 环境没有 PyCUDA 或可用 CUDA，可以临时加 `--backend numpy` 走 CPU fallback；主配置本身保持 CUDA 默认。

采集 oracle episodes：

```bash
/usr/bin/python3 tools/fdm_mppi.py dataset collect \
  --config configs/dataset.yaml \
  --episodes 20 \
  --base-seed 123 \
  --output datasets/oracle_debug
```

构建 split：

```bash
/usr/bin/python3 tools/fdm_mppi.py dataset build \
  --input datasets/oracle_debug \
  --output datasets/oracle_debug_splits
```

校验 dataset：

```bash
/usr/bin/python3 tools/fdm_mppi.py dataset validate \
  --dataset datasets/oracle_debug_splits
```

训练 residual FDM：

```bash
/usr/bin/python3 tools/fdm_mppi.py train \
  --dataset datasets/oracle_debug_splits \
  --output results/fdm_baselines/debug \
  --epochs 50 \
  --device cpu
```

评估 dataset split：

```bash
/usr/bin/python3 tools/fdm_mppi.py eval dataset \
  --dataset datasets/oracle_debug_splits \
  --model-dir results/fdm_baselines/debug \
  --output results/eval_dataset/debug
```

评估 rollout：

```bash
/usr/bin/python3 tools/fdm_mppi.py eval rollout \
  --config configs/smoke.yaml \
  --model-dir results/fdm_baselines/debug \
  --output results/eval_rollout/debug \
  --device cpu
```

运行 closed-loop benchmark：

```bash
/usr/bin/python3 tools/fdm_mppi.py benchmark \
  --config configs/benchmark.yaml \
  --output results/benchmark/debug \
  --episodes 1 \
  --base-seed 123
```

生成最小报告：

```bash
/usr/bin/python3 tools/fdm_mppi.py report \
  --run-path results/sim_results/fdm_mppi_smoke_latest \
  --dataset datasets/oracle_debug_splits \
  --training results/fdm_baselines/debug \
  --benchmark results/benchmark/debug \
  --output results/reports/pipeline_report.json
```

## 输出

仿真输出通常包含：

- `summary.json`
- `experiment_summary.json`（profile 入口生成，集中列出关键产物路径）
- `trajectory.csv`
- `controls.csv`
- `residuals.csv`
- `terrain.csv`
- `animation.gif`（仅在配置开启时生成）

Dataset 输出通常包含：

- `manifest.jsonl`
- `summary.json`
- `train.npz`
- `val.npz`
- `test.npz`
- `split_manifest.json`
- `dataset_summary.json`
- `dataset_quality.json`
- `dataset_summary.png`

`datasets/`、`results/`、checkpoint 和 TensorBoard 输出默认不提交。

## 验证

核心测试：

```bash
/usr/bin/python3 -m pytest -q \
  tests/test_experiment_profile.py \
  tests/test_fdm_mppi_cli.py \
  tests/test_run_omni_mppi_cli.py \
  tests/test_generate_oracle_episodes.py \
  tests/test_oracle_dataset.py \
  tests/test_oracle_dataset_validator.py \
  tests/test_residual_fdm_training.py \
  tests/test_stage5_benchmark.py
```

全量测试：

```bash
/usr/bin/python3 -m pytest -q
```

核心单元测试（Sequence FDM V2）：

```bash
/usr/bin/python3 -m pytest tests/test_sequence_fdm_collector.py -v
```
