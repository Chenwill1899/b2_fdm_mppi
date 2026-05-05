# PR 记录

## 1. 任务目标

新增 NumPy 版 B2 全向 MPPI controller，用于在不改动现有 CUDA 差速 baseline 的前提下，验证 `[vx, vy, wz]` 全向 rollout 和代价计算。

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

- `b2_fdm_mppi/controllers/mppi_omni_numpy.py`
- `tests/test_mppi_omni_numpy.py`

### 修改文件

- `docs/agent_memory/TASK_BOARD.md`
- `docs/agent_memory/EXPERIMENT_LOG.md`
- `docs/agent_memory/GIT_LOG.md`
- `docs/agent_memory/PR_LOG.md`

### 删除文件

- 无

## 4. 算法影响

- [x] 不影响原 baseline
- [ ] 修改动力学模型
- [x] 修改 MPPI 控制器
- [ ] 修改仿真世界模型
- [ ] 修改数据集生成逻辑
- [ ] 修改 FDM 训练逻辑
- [ ] 修改评测指标

## 5. 验证命令

```bash
python3 -m pytest -q
```

保存的本地测试报告：

```text
results/test_reports/20260429_230911/pytest.log
results/test_reports/20260429_230911/pytest.xml
```

验证结果：

```text
25 passed in 3.63s
```

## 6. 关键实现

- 采样候选控制序列 `[K, H, 3]`。
- 使用 `OmniB2` rollout。
- 计算 goal、yaw、control 和 obstacle clearance cost。
- 使用 MPPI 权重更新 nominal control sequence。
- 控制限幅：`vx=1.5`, `vy=0.5`, `wz=1.0`。

## 7. 当前限制与下一步

- 当前 PR 只实现 controller core 和单元测试。
- 下一步需要接入独立 runner/logger，保存 CSV、PNG、GIF 和 summary，然后在 `[18,0]` 双静态障碍场景调参。
