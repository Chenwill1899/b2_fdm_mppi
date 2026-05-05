# Multi-Trajectory Collector — Implementation Plan

**Agent skill:** `superpowers:subagent-driven-development` (or inline execution)

---

## Task 1: Add failing test for multi-trajectory

**File:** `tests/test_sequence_fdm_collector.py`

Add 3 new test functions after existing tests:

```python
def test_collect_multitrajectory_output_structure(tmp_path, monkeypatch):
    """num_trajectories=3 → 3 NPZ files exist, traj_idx 0/1/2."""
    # Same fake_collect_oracle_episode as existing test
    # Call with num_trajectories=3
    # Assert 3 NPZ files exist
    # Assert file names contain traj_00, traj_01, traj_02

def test_collect_multitrajectory_same_terrain(tmp_path, monkeypatch):
    """All 3 files share same terrain_seed."""
    # Same setup
    # Call with num_trajectories=3
    # Load each NPZ, assert terrain_seed identical

def test_collect_multitrajectory_different_start_goal(tmp_path, monkeypatch):
    """Start/goal differ across trajectories."""
    # Same setup
    # Call with num_trajectories=3
    # Load each NPZ, assert start_xy differs between at least 2 files

def test_backward_compat_num_trajectories_1(tmp_path, monkeypatch):
    """num_trajectories=1 default → returns list of 1 dict, single file."""
    # Call collect_sequence_fdm_episode(num_trajectories=1)
    # Assert return is list
    # Assert len(return) == 1
    # Assert single NPZ file exists
```

**Run:** `pytest tests/test_sequence_fdm_collector.py -v`
**Expected:** FAIL — `num_trajectories` param doesn't exist yet

---

## Task 2: Implement multi-trajectory collector

**File:** `b2_fdm_mppi/data/sequence_fdm_collector.py`

Modify `collect_sequence_fdm_episode()`:

### Signature change

```python
def collect_sequence_fdm_episode(
    *,
    # ... existing params ...
    num_trajectories: int = 1,   # NEW, default 1 for backward compat
) -> list[dict]:                 # Changed from dict to list[dict]
```

### Loop structure

```python
def collect_sequence_fdm_episode(*, ..., num_trajectories: int = 1) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = RandomTerrainGenerator(map_bounds=map_bounds, num_patches_range=num_patches_range)
    terrain = generator.generate(seed=terrain_seed)  # generated ONCE at jj=0

    results: list[dict] = []
    for jj in range(num_trajectories):
        # Reseed RNG per trajectory for independent start/goal
        rng = np.random.default_rng(terrain_seed + jj * 1000)
        start_xy, goal_xy = _sample_start_goal(rng, map_bounds, min_distance=min_start_goal_distance)

        # Build config (terrain already generated, re-used)
        config = load_config(base_config_path)
        config["terrain"] = _terrain_to_config(terrain)
        # ... rest of config setup (same as existing) ...
        # Note: mppi seed = terrain_seed + jj so each traj gets different noise realization

        output_path = output_dir / f"episode_{int(episode_id):06d}_traj_{jj:02d}.npz"

        # Write temp YAML, call collect_oracle_episode
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(_convert_tuples_to_lists(config), f)
            temp_config_path = Path(f.name)

        try:
            metadata = collect_oracle_episode(
                config_path=temp_config_path,
                episode_id=episode_id,  # same episode_id for all K
                seed=terrain_seed + jj,
                output_path=output_path,
            )
        finally:
            temp_config_path.unlink(missing_ok=True)

        # Post-process NPZ
        data = dict(np.load(output_path, allow_pickle=True))
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].dtype == object:
                data[key] = data[key].item()

        terrain_risk = data["terrain_risk"]
        binary_risk = _mark_binary_risk(terrain_risk, threshold=risk_threshold)
        data["binary_risk"] = binary_risk
        data["terrain_seed"] = np.asarray(int(terrain_seed), dtype=np.int64)
        data["traj_idx"] = np.asarray(int(jj), dtype=np.int64)
        data["start_xy"] = start_xy.astype(np.float32)
        data["goal_xy"] = goal_xy.astype(np.float32)

        np.savez_compressed(output_path, **data)

        # Build per-trajectory metadata
        metadata["traj_idx"] = jj
        metadata["terrain_seed"] = int(terrain_seed)
        metadata["start_xy"] = start_xy.tolist()
        metadata["goal_xy"] = goal_xy.tolist()
        metadata["output_path"] = str(output_path)
        metadata["success"] = bool(data.get("success", False))
        metadata["final_distance"] = float(data.get("final_distance", float('inf')))
        results.append(metadata)

    return results
```

**Key points:**
- Terrain generated once before the loop (`jj=0`)
- Each trajectory gets independent `start_xy, goal_xy` via reseeded RNG
- Output filename includes `_traj_{jj:02d}` suffix
- `traj_idx` stored in NPZ
- MPPI seed = `terrain_seed + jj` for noise diversity
- Returns `list[dict]` even when `num_trajectories=1`

**Run:** `pytest tests/test_sequence_fdm_collector.py -v`
**Expected:** All 4 new tests + existing tests PASS

---

## Task 3: Update CLI tool

**File:** `tools/collect_sequence_fdm_v2_data.py`

Add `--num-trajectories` argument and flatten manifest results.

### Changes:

```python
parser.add_argument("--num-trajectories", type=int, default=1,
                    help="Number of MPPI trajectories per episode (default: 1)")

# In _collect_one args, add num_trajectories
args = (base_config, episode_id, terrain_seed, output_dir, map_bounds, num_trajectories)

# In main(), update task construction:
tasks = [
    (
        args.base_config,
        episode_id,         # same episode_id for K trajectories
        args.base_seed + i * args.num_trajectories + traj_idx,
        str(output_dir),
        tuple(args.map_bounds),
        args.num_trajectories,
    )
    for i, traj_idx in ...
]
```

Actually simpler: tasks just point to `collect_sequence_fdm_episode` with all params; the collector handles the K loop internally.

```python
# Simpler: single task per episode (collector handles K trajectories internally)
tasks = [
    (args.base_config, i, args.base_seed + i, str(output_dir),
     tuple(args.map_bounds), args.num_trajectories)
    for i in range(args.episodes)
]
```

**Manifest flattening:** When results come back, each result is already `list[dict]` for a single episode. Flatten:

```python
all_trajectories = []
for result_list in results:
    all_trajectories.extend(result_list)

manifest = {
    "results": all_trajectories,
    "summary": {
        "total": len(all_trajectories),
        "episodes": args.episodes,
        "trajectories_per_episode": args.num_trajectories,
        ...
    }
}
```

**Run:** `python tools/collect_sequence_fdm_v2_data.py --help` → should show `--num-trajectories`

---

## Task 4: Verify all tests pass

```bash
pytest tests/test_sequence_fdm_collector.py -v
python tools/collect_sequence_fdm_v2_data.py --help
```

---

## Files Modified

| File | Change |
|------|--------|
| `tests/test_sequence_fdm_collector.py` | Add 4 new test functions |
| `b2_fdm_mppi/data/sequence_fdm_collector.py` | Loop K times, return list[dict] |
| `tools/collect_sequence_fdm_v2_data.py` | Add `--num-trajectories` arg, flatten manifest |

**Constraint:** No changes to `build_sequence_fdm_windows()`, `train_sequence_fdm_v2.py`, or any downstream files.