# Git Log

Last updated: 2026-04-29

## Branch Policy

- `main`: stable version.
- `fdm`: current algorithm development branch.
- `b2-omni-fdm`: B2 omnidirectional and FDM development branch.
- `experiment/*`: concrete experiment branches.
- `fix/*`: bug fix branches.

## Startup State

- Current branch: `fdm`
- Tracking branch: `origin/fdm`
- Working tree status should be checked before every task:

```bash
git status
git branch
```

## Recent Commits

- `8be8e73 fix: make animation non-fatal`
- `c8d73e1 config: save baseline animation gif`
- `7242e23 feat: add baseline summary metrics`
- Pending: `docs: record saved test report`

## Suggested Commit For This Startup Task

```bash
git add docs/agent_memory
git commit -m "docs: initialize agent memory"
```

## Version Tags

Planned tags:

- `v0.1-baseline`
- `v0.2-b2-omni`
- `v0.3-oracle-world`
- `v0.4-fdm-dataset`
- `v0.5-learned-fdm`
- `v0.6-fdm-mppi`
