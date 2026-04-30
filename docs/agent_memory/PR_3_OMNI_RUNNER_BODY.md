# PR 记录

## 1. 任务目标

新增 B2 全向 SE(2) MPPI 仿真 runner，使 Stage 1 能在固定双静态障碍物场景中自动保存 summary、CSV、PNG 和 `animation.gif`，并完成一轮可复现调参。

同时加入 CUDA backend 和 CBF cost：正式 omni 运行现在通过 `mppi.backend: cuda` 使用 PyCUDA rollout/cost kernel，并用 `mppi.cbf_weight` 加入离散 CBF 代价。结果目录改为 `results/sim_results/b2_omni_nominal_<timestamp>/`，避免反复运行覆盖旧结果。

## 2. 所属阶段

- [ ] Baseline 调稳
- [x] B2 全向 SE(2) 模型
- [ ] Oracle Residual World
- [ ] 数据集生成
- [ ] Residual Velocity FDM
- [ ] FDM-MPPI
- [ ] 评测系统
- [ ] 文档/论文整理

## 3. 修改内容

### 新增文件

- `b2_fdm_mppi/simulation/omni_runner.py`
- `b2_fdm_mppi/simulation/results_path.py`
- `b2_fdm_mppi/controllers/mppi_omni_cuda.py`
- `tools/run_omni_mppi.py`
- `tests/test_omni_runner.py`
- `tests/test_mppi_omni_cuda.py`
- `tests/test_results_path.py`

### 修改文件

- `b2_fdm_mppi/controllers/mppi_omni_numpy.py`
- `b2_fdm_mppi/simulation/runner.py`
- `config/b2_omni_nominal.yaml`
- `tests/test_config.py`
- `tests/test_mppi_omni_numpy.py`
- `docs/agent_memory/TASK_BOARD.md`
- `docs/agent_memory/EXPERIMENT_LOG.md`
- `docs/agent_memory/BUG_LOG.md`
- `docs/agent_memory/PR_LOG.md`
- `docs/agent_memory/PR_3_OMNI_RUNNER_BODY.md`

### 删除文件

- 无

## 4. 算法影响

- [x] 不影响原 baseline
- [ ] 修改动力学模型
- [x] 修改 MPPI 控制器
- [ ] 修改仿真世界模型
- [ ] 修改数据集生成逻辑
- [ ] 修改 FDM 训练逻辑
- [x] 修改评测指标

## 5. 验证命令

```bash
python3 -m pytest -q
python3 tools/run_omni_mppi.py --config config/b2_omni_nominal.yaml --seed 123
```

## 6. 验证结果

```text
39 passed in 2.46s
```

实际仿真结果：

```text
results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/
```

关键指标：

```text
success: true
steps: 142
final_distance: 0.3754442036151886
path_length: 18.206396102905273
arrival_time: 14.200000000000001
mean_mppi_time_ms: 4.929683577846474
max_mppi_time_ms: 7.981300354003906
min_obstacle_clearance: 0.4295613765716553
```

关键输出：

```text
results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/animation.gif
results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/trajectory.png
results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/summary.json
results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/trajectory.csv
results/sim_results/b2_omni_nominal_2026-04-30_13-54-40/controls.csv
```

Result directory check:

```text
results_path=results/sim_results/b2_omni_nominal_2026-04-30_13-54-40
```

## 7. 当前限制与下一步

- 当前 runner 仍是 nominal world，不包含 oracle residual world。
- 当前 CBF 是 CUDA cost penalty，不是完整 soft/slack RCBF。
- `animation.gif` 已恢复 sampled candidate rollouts 和 optimized rollout 显示。
- `smooth_weight=2.0` 能改善轨迹平滑性，但后续 Stage 2/3 仍需要正式加入 `control_smoothness` 指标。
- 旧时间戳结果目录未自动删除，避免误删历史检查材料。
- 下一步进入 Oracle Residual World：新增 `core/terrain.py` 和 `core/residual_world.py`，保持 nominal runner 不破坏。
