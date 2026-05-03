# [MPPI] refine static-obstacle nominal planner smoothness

## 1. 任务目标

Stage 1.5 清理静态障碍 MPPI baseline，关闭默认 RCBF 依赖，补齐平滑性指标，并改善轨迹平滑性。

## 2. 修改内容

- 修改 `config/b2_omni_nominal.yaml`
  - 默认关闭 RCBF：`cbf.enabled: false`，`cbf.type: 0`，`mppi.cbf_weight: 0.0`
  - 保留静态障碍距离代价：`mppi.obstacle_weight: 300.0`
  - 提高静态安全裕度：`robot.safety_dist: 0.5`
  - 启用执行端低通滤波：`execution.filter_enabled: true`，`execution.filter_alpha: 0.6`
- 修改 `b2_fdm_mppi/simulation/omni_runner.py`
  - 新增 executed control smoothness / jerk / variance summary 指标
  - `controls.csv` 保存 executed controls
  - 新增 `raw_controls.csv` 保存滤波前 MPPI controls
- 修改 `b2_fdm_mppi/controllers/mppi_omni_cuda.py`
  - `cbf.enabled: false` 时强制 `cbf_weight=0.0`
- 更新测试
  - `tests/test_omni_runner.py`
  - `tests/test_mppi_omni_cuda.py`
  - `tests/test_config.py`
  - `tests/test_mppi_omni_numpy.py`
- 更新 agent memory
  - `docs/agent_memory/TASK_BOARD.md`
  - `docs/agent_memory/EXPERIMENT_LOG.md`
  - `docs/agent_memory/PR_LOG.md`

## 3. 为什么关闭默认 RCBF

当前 RCBF 更适合动态障碍。FDM 第一阶段只考虑静态障碍，因此 Stage 1.5 默认使用静态障碍距离代价。RCBF 代码保留为后续动态障碍扩展，但不作为当前静态 FDM 主线依赖。

## 4. 验证命令

```bash
python3 -m pytest -q
python3 tools/run_omni_mppi.py --config config/b2_omni_nominal.yaml --seed 123
```

## 5. 旧结果

Result:

```text
results/sim_results/b2_omni_nominal_2026-04-30_14-45-30/
```

Metrics:

```text
success: true
final_distance: 0.34110841155052185
path_length: 18.318750381469727
mean_mppi_time_ms: 4.219803545210096
max_mppi_time_ms: 9.484291076660156
min_obstacle_clearance: 0.4714846611022949
control_smoothness: not_available in summary
control_jerk: not_available in summary
computed_control_smoothness_from_controls_csv: 0.1314503344222518
computed_control_jerk_from_controls_csv: 0.36088919826191274
```

## 6. 新结果

Result:

```text
results/sim_results/b2_omni_nominal_2026-04-30_14-52-40/
```

Metrics:

```text
success: true
final_distance: 0.3399098217487335
path_length: 18.460805892944336
arrival_time: 16.2
mean_mppi_time_ms: 5.112684803244508
max_mppi_time_ms: 16.221046447753906
min_obstacle_clearance: 0.4645106792449951
control_smoothness: 0.012281207671864226
control_jerk: 0.020258904777513565
smooth_vx: 0.0033688947038119903
smooth_vy: 0.002820520426709939
smooth_wz: 0.006091792541342289
jerk_vx: 0.003348826443667603
jerk_vy: 0.005618836767139562
jerk_wz: 0.011291241566706387
vx_variance: 0.1710001605578116
vy_variance: 0.018017247619684575
wz_variance: 0.03171373441428392
```

Artifacts:

```text
summary.json
trajectory.csv
controls.csv
raw_controls.csv
trajectory.png
animation.gif
```

## 7. 对比结论

- 仍能到达目标：`success=true`
- 终点误差保持达标：`0.3399 m < 0.45 m`
- 安全距离保持达标：`0.4645 m > 0.4 m`
- 计算时间可接受：`mean_mppi_time_ms=5.1127 ms < 20 ms`
- `control_smoothness` 从按旧 `controls.csv` 计算的 `0.13145` 降到 `0.01228`
- `control_jerk` 从按旧 `controls.csv` 计算的 `0.36089` 降到 `0.02026`
- 默认配置不依赖动态 RCBF

## 8. 当前限制

- 当前仍是静态障碍 baseline
- RCBF 未作为主线，仅保留为动态障碍扩展
- 当前未引入 FDM
- 当前未构建 oracle residual world

## 9. 下一步

进入 Oracle Residual World：

- add terrain field
- add residual world
- run nominal MPPI in oracle world
- generate residual metrics
