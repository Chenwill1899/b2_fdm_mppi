# B2-FDM-MPPI Project Context

Last updated: 2026-05-01

## Repository

- Path: `/home/mexxiie/prj/py-mppi`
- Active development branch: `dev`
- Package: ROS 2 Humble `ament_python` package `b2_fdm_mppi`
- Current baseline structure: PyCUDA MPPI/CBF internal simulation with Jackal-style differential drive model.

## Notion Sync

- Project cockpit: https://www.notion.so/fdm_mppi-3532fb2e849f8188a4fef5bb3ae54264?t=3532fb2e849f8080ac1400a9065381ce
- Every completed small stage must be updated in Notion before closing the task.
- Notion sync must update both the relevant database records and the main project page progress when project-visible state changes.
- Use the Notion databases by content type: tasks in `fdm_mppi 任务看板`, experiments in `fdm_mppi 实验记录`, risks in `fdm_mppi 风险与问题`, decisions in `fdm_mppi 决策日志`, daily summaries in `fdm_mppi 日报记录`.
- Main page progress updates include current stage, stage focus checklist, roadmap status, latest stable conclusion, and next step.
- Stage updates should include commands, configs, backend, seeds, output paths, key metrics/results, changed files, verification commands, conclusion, and next step when applicable.
- If Notion access is unavailable, update the local `docs/agent_memory/` files and leave a pending Notion-sync note.

## User Workflow Rule

- Always make code changes on the local `dev` branch only.
- Do not create or continue feature branches unless the user explicitly asks for one.
- After completing and verifying changes, push `dev`; the user will handle merging.
- Before any code edit, check the branch with `git branch --show-current` and switch to `dev` if needed.

## Research Goal

Build a reproducible numerical simulation and evaluation platform for B2 quadruped residual velocity FDM-MPPI.

Core chain:

1. Use an omnidirectional SE(2) model as the nominal dynamics.
2. Build an oracle residual world to emulate B2 execution bias.
3. Generate oracle residual datasets.
4. Train a residual velocity forward dynamics model (FDM).
5. Use the learned FDM as the MPPI rollout model.
6. Compare Nominal-MPPI, Oracle-MPPI, Learned-FDM-MPPI, and risk-aware variants.

## Important Distinction

- FDM is a planning-level black-box closed-loop forward dynamics model.
- MPPI is the sampling-based planning/control framework that uses the model.
- The project does not replace full quadruped analytical dynamics. It learns execution residuals on top of a nominal omnidirectional SE(2) model.

## Current Baseline Facts

- Default config: `config/fdm_mppi.yaml`
- Default target: `[10.0, 0.0, 0.0, 0.0, 0.0]`
- Default plots and animation are enabled.
- Results currently save CSV files and `test_summary.yaml` under `results/sim_results/<timestamp>/`.
- Existing tests use a fake controller for non-GPU runner coverage.
