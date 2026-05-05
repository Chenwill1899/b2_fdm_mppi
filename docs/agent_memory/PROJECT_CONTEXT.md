# B2-FDM-MPPI Project Context

Last updated: 2026-05-03

## Repository

- Path: `/home/mexxiie/prj/py-mppi`
- Active refactor branch: `new`
- Package: ROS 2 Humble `ament_python` package `b2_fdm_mppi`
- Current structure: slim FDM-MPPI reproducible pipeline with one recommended CLI and archived historical stage assets.

## Notion Sync

- Project cockpit: https://www.notion.so/fdm_mppi-3532fb2e849f8188a4fef5bb3ae54264?t=3532fb2e849f8080ac1400a9065381ce
- Every completed small stage must be updated in Notion before closing the task.
- Notion sync must update both the relevant database records and the main project page progress when project-visible state changes.
- Use the Notion databases by content type: tasks in `fdm_mppi 任务看板`, experiments in `fdm_mppi 实验记录`, risks in `fdm_mppi 风险与问题`, decisions in `fdm_mppi 决策日志`, daily summaries in `fdm_mppi 日报记录`.
- Main page progress updates include current stage, stage focus checklist, roadmap status, latest stable conclusion, and next step.
- Stage updates should include commands, configs, backend, seeds, output paths, key metrics/results, changed files, verification commands, conclusion, and next step when applicable.
- If Notion access is unavailable, update the local `docs/agent_memory/` files and leave a pending Notion-sync note.

## Current Route

- Current requested branch: `new`.
- Source branch synced into `new`: latest `fdm` as of 2026-05-03.
- Main entry point: `python3 tools/fdm_mppi.py`.
- Recommended configs:
  - `configs/smoke.yaml`
  - `configs/dataset.yaml`
  - `configs/benchmark.yaml`
- Historical Stage 5 / Stage 6 scripts and published artifacts are under `archive/`.
- Old `tools/*.py` and `config/*.yaml` paths remain as compatibility anchors, not the default workflow.

## Research Goal

Build a reproducible numerical simulation and evaluation platform for B2 quadruped residual velocity FDM-MPPI.

Slim core chain:

1. Use an omnidirectional SE(2) model as the nominal dynamics.
2. Build an oracle residual world to emulate B2 execution bias.
3. Generate oracle residual datasets.
4. Train a residual velocity forward dynamics model (FDM).
5. Use the learned FDM as the MPPI rollout model.
6. Evaluate closed-loop / rollout behavior and write a compact report.

## Important Distinction

- FDM is a planning-level black-box closed-loop forward dynamics model.
- MPPI is the sampling-based planning/control framework that uses the model.
- The project does not replace full quadruped analytical dynamics. It learns execution residuals on top of a nominal omnidirectional SE(2) model.

## Current Pipeline Facts

- Smoke config defaults to NumPy backend and writes to `results/sim_results/fdm_mppi_smoke_latest`.
- Dataset config defaults to NumPy backend and keeps `results.enable_plots=false`, `results.enable_animation=false`.
- Training target remains executed residuals: `exec_residuals = real_controls - cmd_controls`.
- Generated `datasets/` and `results/` artifacts are verification outputs and should not be committed by default.
