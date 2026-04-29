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
- `f6bb68d docs: record saved test report`
- `61a148a docs: record saved gif artifact`
- `74afe85 config: add short-goal baseline`
- `7f4be5d config: add straight obstacle baseline`
- `e7866b1 fix: keep static obstacles stationary`
- `e890487 config: set double static obstacle scene`
- `0aa3dba config: update double obstacle positions`
- `3baeb81 feat: add b2 omni nominal model`
- `9b7bdce docs: add pr log for b2 omni model`
- `c1fe15b docs: record b2 omni pr push blocker`
- Pending: `docs: record b2 omni pr created`

## Push Blockers

- `2026-04-29`: `git push -u origin feature/b2-omni-model` failed because local GitHub HTTPS credentials were unavailable:

```text
fatal: could not read Username for 'https://github.com': 没有那个设备或地址
```

- `2026-04-29`: blocker resolved after `gh` authentication was restored. Branch `feature/b2-omni-model` was pushed and PR #1 was created:

```text
https://github.com/Chenwill1899/b2_fdm_mppi/pull/1
```

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
