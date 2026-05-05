# Stage 3 Seeding Plan

Last updated: 2026-04-30

## Goal

Stage 3 parallel oracle dataset generation must make every episode reproducible from `episode_id`.

The seed contract should be explicit:

```text
episode_id -> scenario_seed / oracle_seed / obstacle_seed / terrain_seed
```

Do not rely on process-local randomness inside parallel workers for final dataset generation.

## Modes

### fixed_map_random_tasks

Use this mode for the current Stage 2.6 dataset-friendly config:

```text
config/b2_omni_oracle_random100_dataset.yaml
```

Semantics:

- Fixed terrain.
- Fixed random obstacle map.
- Fixed oracle residual noise seed unless intentionally varied.
- Random start/goal task per episode.

Current config behavior:

```yaml
scenario:
  random_seed: auto

obstacles:
  random_seed: 123

terrain:
  noise_seed: 123

oracle_residual:
  seed: 123
```

This is suitable for quick single-run inspection because `scenario.random_seed: auto` changes start/goal each run while map seeds stay fixed.

For Stage 3 parallel collection, replace `auto` at runtime with an episode-derived integer:

```text
scenario_seed = base_seed + episode_id
obstacle_seed = 123
terrain_seed = 123
oracle_seed = 123
```

The saved per-episode `config.yaml` and `summary.json` must contain the resolved integer seeds.

### random_map_random_tasks

Use this mode when the dataset should cover many terrain and obstacle maps.

Recommended deterministic mapping:

```text
scenario_seed = base_seed + episode_id
obstacle_seed = base_seed + 100000 + episode_id
terrain_seed = base_seed + 200000 + episode_id
oracle_seed = base_seed + 300000 + episode_id
```

Semantics:

- Terrain changes per episode.
- Obstacle map changes per episode.
- Start/goal changes per episode.
- Oracle residual noise changes per episode.

## Stage 3 Implementation Requirement

The parallel collector should accept:

```text
mode
base_seed
episode_id range
num_workers
```

Each worker must write resolved seeds into output artifacts before running the episode. This keeps failed episodes debuggable and allows exact replay by loading the saved config.
