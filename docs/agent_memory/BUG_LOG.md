# Bug Log

Last updated: 2026-04-29

## Known Risks

| ID | Status | Symptom | Likely Cause | Next Action |
| --- | --- | --- | --- | --- |
| B-001 | fixed | Animation may fail or block run completion | ffmpeg or matplotlib animation writer unavailable; runner previously let animation exceptions propagate | Default animation disabled in `config/fdm_mppi.yaml`; runner catches animation exceptions and emits `RuntimeWarning`. Verified by `python3 -m pytest -q` on 2026-04-29. |
| B-002 | open | Real MPPI controller may fail to import/run | PyCUDA/NVIDIA runtime unavailable | Keep fake-controller tests as CPU smoke coverage; verify GPU separately. |
| B-003 | open | Summary lacks Stage 0 acceptance metrics | Current `test_summary.yaml` only stores init, goal, steps, failed, average time | Add final distance, success, path length, max time, and run time. |

## Fixed Bugs

- B-001: animation failure is non-fatal after `runner._plot_results()` wraps `utils.animate_simulation()`.
