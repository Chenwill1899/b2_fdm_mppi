# Git Log

Last updated: 2026-04-30

## Branch Policy

- User rule: all assistant code changes must happen on `dev` only.
- Push completed verified work to `origin/dev`; the user will merge.
- Do not create or continue feature branches unless explicitly requested.
- Check the current branch before editing:

```bash
git branch --show-current
git status --short --branch
```

## Startup State

- Expected current branch for work: `dev`
- Tracking branch: `origin/dev`
- Working tree status should be checked before every task:

```bash
git status
git branch
```

## Recent Commits

- `feat: refine kinodynamic MPPI baseline` planned on `dev` for Stage 1.5 final baseline split and far-field static obstacle potential.
- `feat: add kinodynamic omni rollout` on `dev`.
- `2a085a8 feat: refine static obstacle MPPI smoothness metrics` cherry-picked onto `dev` from the accidental feature branch.
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
- `084d236 docs: record b2 omni pr created`
- `8e2b14e feat: add numpy omni mppi controller`
- `d8895ee docs: add pr2 body for numpy omni mppi`
- `feat: add omni mppi runner` on `feature/omni-mppi-runner`

## Current PR Stack

- PR #1: `feature/b2-omni-model` -> `fdm`
  - `https://github.com/Chenwill1899/b2_fdm_mppi/pull/1`
- PR #2: `feature/omni-mppi-numpy` -> `feature/b2-omni-model`
  - `https://github.com/Chenwill1899/b2_fdm_mppi/pull/2`
- PR #3: `feature/omni-mppi-runner` -> `feature/omni-mppi-numpy`
  - `https://github.com/Chenwill1899/b2_fdm_mppi/pull/3`
  - local commit: `feat: add omni mppi runner`
  - verified by `results/test_reports/20260429_232717/`
  - real run: `results/sim_results/2026-04-29_23-27-26/`

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
