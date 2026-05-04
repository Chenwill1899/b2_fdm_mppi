# Multi-Trajectory Sequence FDM Collector Design

**Date**: 2026-05-04
**Status**: Approved
**Scope**: Single feature — collect K>1 MPPI trajectories per episode, terrain generated once, each trajectory with independent start/goal

---

## 1. Problem

`collect_sequence_fdm_episode()` currently generates one random terrain + runs one MPPI trajectory per call. To increase data efficiency (terrain construction cost shared across trajectories) and diversity (same terrain but different goals), we extend the collector to support `num_trajectories > 1`.

---

## 2. Design

### 2.1 API

```python
def collect_sequence_fdm_episode(
    *,
    base_config_path: str | Path,
    episode_id: int,
    terrain_seed: int,
    output_dir: str | Path,
    num_trajectories: int = 1,          # NEW
    map_bounds: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
    min_start_goal_distance: float = 10.0,
    risk_threshold: float = 0.6,
    num_patches_range: tuple[int, int] = (3, 6),
) -> list[dict]:                         # Changed from dict to list[dict]
```

**Backward compatible:** `num_trajectories=1` (default) returns `[{...}]` — a list of one element — preserving existing callers that unpack `.episode_id` etc.

### 2.2 Per-Trajectory Flow (K trajectories)

```
for jj in range(K):
    1. if jj == 0: generate terrain + obstacles once
    2. sample start_xy, goal_xy independently (rng reseeded per trajectory)
    3. override config["simulation"]["initial_state"] and config["simulation"]["goal"]
    4. write temp YAML
    5. collect_oracle_episode(config_path=temp_yaml, episode_id=episode_id, seed=terrain_seed + jj, output_path=...)
    6. post-process NPZ: add binary_risk, terrain_seed, traj_idx=jj, start_xy, goal_xy
```

**Key:** Terrain generated once at `jj=0`. MPPI seed offsets by `+jj` so each trajectory sees slightly different noise realization.

### 2.3 Output Files

```
{output_dir}/
├── episode_{episode_id:06d}_traj_00.npz   ← trajectory 0
├── episode_{episode_id:06d}_traj_01.npz   ← trajectory 1
├── ...
├── episode_{episode_id:06d}_traj_{K-1:02d}.npz
└── manifest.json                            ← flat list of all trajectories
```

NPZ fields per file:
- `states`: (T, 6) float32
- `cmd_controls`: (T, 3) float32
- `binary_risk`: (T,) float32
- `terrain_seed`: int64 (shared across K files)
- `traj_idx`: int64 (0…K-1, unique per file)
- `start_xy`: (2,) float32
- `goal_xy`: (2,) float32

### 2.4 Failure Handling

- If MPPI crashes on trajectory `jj`: log error in manifest for that trajectory, continue with remaining trajectories
- `success` field in metadata reflects that individual trajectory's outcome
- **Failure trajectories are kept** — they contribute valuable negative training samples (binary_risk=1 regions)

### 2.5 Start/Goal Sampling

Each trajectory independently calls `_sample_start_goal(rng, map_bounds, min_distance=10.0)`.
Rng is derived from `np.random.default_rng(terrain_seed + jj * 1000)` to keep reproducibility but ensure diversity.

---

## 3. Files Modified

| File | Change |
|------|--------|
| `b2_fdm_mppi/data/sequence_fdm_collector.py` | `collect_sequence_fdm_episode`: loop over K, return `list[dict]` |
| `tools/collect_sequence_fdm_v2_data.py` | Add `--num-trajectories` arg; flatten manifest |

---

## 4. Test Plan

1. `test_collect_multitrajectory_output_structure`: `num_trajectories=3` → 3 NPZ files exist, each with `traj_idx` 0/1/2
2. `test_collect_multitrajectory_same_terrain`: all 3 files share same `terrain_seed`
3. `test_collect_multitrajectory_different_start_goal`: start/goal differ across trajectories
4. `test_backward_compat_num_trajectories_1`: default param → list of 1 dict returned, single file
5. CLI smoke test with `--num-trajectories 2`

---

## 5. No Changes to Downstream

- `build_sequence_fdm_windows()` — unchanged, processes individual NPZ files as before
- `train_sequence_fdm_v2.py` — unchanged, `glob("episode_*.npz")` matches `_traj_*.npz` files too (lexicographic sort gives correct ordering)
- Iteration training (`--resume`) — **out of scope** for this spec, handled separately