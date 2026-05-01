# Repository Agent Rules

This project is linked to the Notion project cockpit:

https://www.notion.so/fdm_mppi-3532fb2e849f8188a4fef5bb3ae54264?t=3532fb2e849f8080ac1400a9065381ce

When a small stage is completed, update Notion before closing the task. A small stage includes items like `S4-003`, a dataset generation run, a training benchmark, a validation pass, a bugfix that unblocks a stage, or a stage transition.

Notion updates must cover both:

- The relevant Notion database entry or entries.
- The main project page progress when the stage changes project-visible state, including current stage, stage focus checklist, roadmap status, latest stable conclusion, or next step.

Record the update in the relevant Notion database:

- Task / todo / blocker / next step: `fdm_mppi 任务看板`
- Training / evaluation / dataset / seed / metric / command / output path: `fdm_mppi 实验记录`
- Bug / risk / uncertainty / stage blocker: `fdm_mppi 风险与问题`
- Route change / technical decision / stage transition / important conclusion: `fdm_mppi 决策日志`
- Daily progress summary / next-day plan / short retrospective: `fdm_mppi 日报记录`

For experiment or dataset stages, include the command, config, backend, seed mapping, output path, key metrics, changed files, verification command, and conclusion. If Notion access is unavailable, update `docs/agent_memory/` locally and leave a clear pending Notion-sync note.
