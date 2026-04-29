# PR 记录

## 1. 任务目标

新增 B2 全向 SE(2) NumPy MPPI 仿真 runner，使 Stage 1 能在固定双静态障碍物场景中自动保存 summary、CSV、PNG 和 `animation.gif`，并完成一轮可复现调参。

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
- `tools/run_omni_mppi.py`
- `tests/test_omni_runner.py`

### 修改文件

- `b2_fdm_mppi/controllers/mppi_omni_numpy.py`
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
python3 -m pytest tests/ -v --junitxml=results/test_reports/20260429_232717/pytest.xml
python3 tools/run_omni_mppi.py --config config/b2_omni_nominal.yaml --seed 123
```

## 6. 验证结果

```text
33 passed in 1.99s
```

测试报告：

```text
results/test_reports/20260429_232717/pytest.log
results/test_reports/20260429_232717/pytest.xml
```

实际仿真结果：

```text
results/sim_results/2026-04-29_23-27-26/
```

关键指标：

```text
success: true
steps: 134
final_distance: 0.36341118812561035
path_length: 18.1884765625
arrival_time: 13.4
mean_mppi_time_ms: 6.374088685903976
max_mppi_time_ms: 18.85843276977539
min_obstacle_clearance: 0.38778746128082275
```

关键输出：

```text
results/sim_results/2026-04-29_23-27-26/animation.gif
results/sim_results/2026-04-29_23-27-26/trajectory.png
results/sim_results/2026-04-29_23-27-26/summary.json
results/sim_results/2026-04-29_23-27-26/trajectory.csv
results/sim_results/2026-04-29_23-27-26/controls.csv
```

## 7. 当前限制与下一步

- 当前 runner 仍是 nominal world，不包含 oracle residual world。
- `animation.gif` 已恢复 sampled candidate rollouts 和 optimized rollout 显示。
- `smooth_weight=1.0` 能改善轨迹平滑性，但后续 Stage 2/3 仍需要正式加入 `control_smoothness` 指标。
- 下一步进入 Oracle Residual World：新增 `core/terrain.py` 和 `core/residual_world.py`，保持 nominal runner 不破坏。
