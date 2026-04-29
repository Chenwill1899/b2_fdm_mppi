# PR Log

Last updated: 2026-04-29

## PR #1: B2 Omni Nominal Model

- Repository: `Chenwill1899/b2_fdm_mppi`
- PR: `https://github.com/Chenwill1899/b2_fdm_mppi/pull/1`
- State: `OPEN`
- Base branch: `fdm`
- Head branch: `feature/b2-omni-model`
- Title: `[B2-Omni] feat: add SE(2) omnidirectional B2 model`
- Local commits:
  - `c1fe15b docs: record b2 omni pr push blocker`
  - `9b7bdce docs: add pr log for b2 omni model`
  - `3baeb81 feat: add b2 omni nominal model`
- Upload status: pushed and PR created after `gh` authentication was restored.

## Upload History

- `2026-04-29`: first push attempt failed because local GitHub HTTPS credentials were unavailable:

```text
fatal: could not read Username for 'https://github.com': 没有那个设备或地址
```

- `2026-04-29`: `gh auth status` passed as user `Chenwill1899`.
- `2026-04-29`: branch pushed:

```bash
git push -u origin feature/b2-omni-model
```

- `2026-04-29`: PR created:

```text
https://github.com/Chenwill1899/b2_fdm_mppi/pull/1
```

## PR Body

```markdown
# PR 记录

## 1. 任务目标

新增 B2 全向 SE(2) 名义动力学模型，为后续全向 MPPI rollout、oracle residual world 和 residual velocity FDM 接入做基础。

本 PR 只完成模型与配置的可验证基础层，不替换现有差速 CUDA MPPI baseline。

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

- `b2_fdm_mppi/core/omni_b2.py`
- `config/b2_omni_nominal.yaml`

### 修改文件

- `b2_fdm_mppi/config.py`
- `tests/test_models.py`
- `tests/test_config.py`
- `docs/agent_memory/TASK_BOARD.md`
- `docs/agent_memory/EXPERIMENT_LOG.md`
- `docs/agent_memory/FDM_DESIGN.md`
- `docs/agent_memory/GIT_LOG.md`

### 删除文件

- 无

## 4. 算法影响

- [x] 不影响原 baseline
- [x] 修改动力学模型
- [ ] 修改 MPPI 控制器
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
results/test_reports/20260429_223738/pytest.log
results/test_reports/20260429_223738/pytest.xml
```

验证结果：

```text
20 passed in 1.03s
```

## 6. 关键实现

- 新增 `OmniB2` 名义模型。
- 状态定义：

```text
[x, y, theta, vx_real, vy_real, wz_real]
```

- 控制定义：

```text
[vx_cmd, vy_cmd, wz_cmd]
```

- 运动学：

```text
x_next     = x + (vx cos(theta) - vy sin(theta)) dt
y_next     = y + (vx sin(theta) + vy cos(theta)) dt
theta_next = theta + wz dt
```

- 速度限幅：

```text
max_vx = 1.5
max_vy = 0.5
max_wz = 1.0
```

## 7. 当前限制与下一步

- 当前 CUDA MPPI controller 仍是差速 `[v, w]`，本 PR 不直接替换它。
- 下一步新增 `controllers/mppi_omni_numpy.py`，先用 NumPy 验证全向 MPPI rollout，再进入调参。
```

## Local Verification Artifacts

- `results/test_reports/20260429_223738/pytest.log`
- `results/test_reports/20260429_223738/pytest.xml`

## PR Upload Status

- `2026-04-29`: local branch `feature/b2-omni-model` created.
- `2026-04-29`: push attempted with `git push -u origin feature/b2-omni-model`.
- `2026-04-29`: push blocked by missing local GitHub HTTPS credentials.
- `2026-04-29`: push succeeded after `gh` was restored.
- `2026-04-29`: PR #1 opened at `https://github.com/Chenwill1899/b2_fdm_mppi/pull/1`.

## PR Candidate: NumPy Omni MPPI

- Repository: `Chenwill1899/b2_fdm_mppi`
- PR: `https://github.com/Chenwill1899/b2_fdm_mppi/pull/2`
- State: `OPEN`
- Base branch: `feature/b2-omni-model`
- Head branch: `feature/omni-mppi-numpy`
- Title: `[MPPI] feat: add NumPy omnidirectional MPPI controller`
- Current verification:

```text
results/test_reports/20260429_230911/pytest.log
results/test_reports/20260429_230911/pytest.xml
25 passed in 3.63s
```

- Scope:
  - Add `b2_fdm_mppi/controllers/mppi_omni_numpy.py`
  - Add tests for control limits, goal progress, obstacle cost, config construction, and closed-loop smoke behavior.
  - Keep existing CUDA differential MPPI unchanged.
- Body file: `docs/agent_memory/PR_2_OMNI_MPPI_BODY.md`

## PR Candidate: Omni MPPI Runner and Tuned Scene

- Repository: `Chenwill1899/b2_fdm_mppi`
- Base branch: `feature/omni-mppi-numpy`
- Head branch: `feature/omni-mppi-runner`
- Title: `[MPPI] feat: add omni MPPI runner and tuning outputs`
- Current verification:

```text
results/test_reports/20260429_232717/pytest.log
results/test_reports/20260429_232717/pytest.xml
33 passed in 1.99s
```

- Real run:

```text
results/sim_results/2026-04-29_23-27-26/
success: true
final_distance: 0.36341118812561035
mean_mppi_time_ms: 6.374088685903976
min_obstacle_clearance: 0.38778746128082275
animation.gif: saved, 680K
```

- Scope:
  - Add B2 omni simulation runner and CLI.
  - Save `summary.json`, `test_summary.yaml`, `trajectory.csv`, `controls.csv`, `obs_results.csv`, `time_results.csv`, `costs.csv`, `trajectory.png`, and `animation.gif`.
  - Restore sampled candidate rollout and optimized rollout display in GIF.
  - Tune harder double-obstacle scene with `obstacle_weight=800`, `safety_dist=0.4`, and `smooth_weight=1.0`.
- Body file: `docs/agent_memory/PR_3_OMNI_RUNNER_BODY.md`
