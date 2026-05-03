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
- `b2_fdm_mppi/data/`: oracle episode、dataset split、dataset validator。
- `b2_fdm_mppi/training/`: residual FDM 训练。
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

旧 `config/*.yaml` 仍可用于复现实验和兼容测试，但主流程不再依赖这些路径。

## 常用命令

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
  --model-dir results/fdm_baselines/debug \
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

每次 experiment 会写出 `experiment_summary.json`，里面列出 `summary.json`、`trajectory.csv`、`trajectory.png`、`oracle_diagnostics.png`、`animation.gif` 等可检查产物路径。

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
