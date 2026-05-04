# Iterative Training Design

**Date**: 2026-05-04
**Status**: Approved
**Scope**: 3-round iterative training loop: collect → train → repeat, with `--resume` checkpoint support

---

## 1. Problem

Training currently does a single round: collect episodes → extract windows → train curriculum → done. To improve model quality iteratively, we need:
1. **Resume support** — continue training from a checkpoint (model + optimizer state)
2. **Iterative loop** — 3 rounds where each round collects new data and continues training

---

## 2. Resume Support

### 2.1 Training Function

```python
def train_sequence_fdm_v2(
    windows: list[dict],
    output_dir: str | Path,
    resume_from: str | Path | None = None,   # NEW: checkpoint to resume from
    curriculum_phases: list[tuple[int, int, float]] | None = None,
    ...
) -> dict[str, Any]:
```

When `resume_from` is provided:
- Load `best_model.pt` from `resume_from` directory
- Create model with matching `horizon_steps` and `hidden_dims`
- Load model weights into the **last phase's model** (highest horizon)
- Resume optimizer from saved `optimizer.pt` state (if exists)
- Continue curriculum from the last phase
- **Normalization is recomputed** on new data (old normalization is discarded)

### 2.2 Checkpoint Structure

```
{output_dir}/
├── best_model.pt          # model weights + metadata
├── best_model_h{horizon}.pt   # per-phase best model
├── optimizer.pt           # optimizer state (for resume)
├── normalization.npz      # normalization params
├── training_metrics.json  # full training history
└── round_{N}/            # per-round subdirectory
    ├── best_model.pt
    ├── best_model_h5.pt, best_model_h10.pt, best_model_h20.pt
    └── training_metrics.json
```

### 2.3 CLI Changes

```bash
python tools/train_sequence_fdm_v2.py \
  --data-dir data/round1 \
  --output-dir checkpoints/round2 \
  --resume checkpoints/round1       # load from previous round
```

---

## 3. Iterative Loop (3 Rounds)

### 3.1 Round Configuration

| Round | Episodes (base) | Trajectories | Total Trajectories | Training |
|-------|-----------------|-------------|-------------------|----------|
| 1 | 100 | 3 | 300 | Full curriculum H=5→10→20 |
| 2 | 100 | 3 | 300 (new) | Resume from round1, H=5→10→20 |
| 3 | 200 | 3 | 600 (new) | Resume from round2, H=5→10→20 |

**Note:** Each round's data-dir contains **only that round's new episodes**. The resume mechanism loads the previous checkpoint, but the data directory contains only fresh data (incremental, not cumulative files).

### 3.2 Data Collection Per Round

```bash
# Round 1
python tools/collect_sequence_fdm_v2_data.py \
  --base-config configs/smoke.yaml \
  --episodes 100 \
  --num-trajectories 3 \
  --output-dir data/round1 \
  --base-seed 0

# Round 2
python tools/collect_sequence_fdm_v2_data.py \
  --base-config configs/smoke.yaml \
  --episodes 100 \
  --num-trajectories 3 \
  --output-dir data/round2 \
  --base-seed 100000    # different seed range from round 1

# Round 3
python tools/collect_sequence_fdm_v2_data.py \
  --base-config configs/smoke.yaml \
  --episodes 200 \
  --num-trajectories 3 \
  --output-dir data/round3 \
  --base-seed 200000
```

### 3.3 Training Per Round

```bash
# Round 1 — no resume
python tools/train_sequence_fdm_v2.py \
  --data-dir data/round1 \
  --output-dir checkpoints/round1

# Round 2 — resume from round 1
python tools/train_sequence_fdm_v2.py \
  --data-dir data/round2 \
  --output-dir checkpoints/round2 \
  --resume checkpoints/round1

# Round 3 — resume from round 2
python tools/train_sequence_fdm_v2.py \
  --data-dir data/round3 \
  --output-dir checkpoints/round3 \
  --resume checkpoints/round2
```

---

## 4. Files Modified

| File | Change |
|------|--------|
| `b2_fdm_mppi/training/sequence_fdm_v2.py` | Add `resume_from` param; save optimizer state; load model + optimizer on resume |
| `tools/train_sequence_fdm_v2.py` | Add `--resume` CLI arg; pass to training function |
| `tools/run_iterative_training.py` | NEW: orchestration script for 3-round iterative loop |

---

## 5. No Changes to Downstream

- `build_sequence_fdm_windows()` — unchanged
- `collect_sequence_fdm_v2_data.py` — unchanged
- Dataset builder — unchanged