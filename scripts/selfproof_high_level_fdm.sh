#!/usr/bin/env bash
# End-to-end self-proof for the High-Level FDM.
#
# Produces a tiny synthetic dataset, trains for a few epochs on CPU,
# evaluates, runs the MPPI rollout contract check, and writes a single
# machine-readable JSON with PASS/FAIL verdicts.
#
# Usage:
#   scripts/selfproof_high_level_fdm.sh [OUTPUT_ROOT]
#
# Requirements in the caller's environment:
#   python3 with numpy, torch, pyyaml
#   pytest (optional, only used if you want the unit suite at the end)

set -euo pipefail

OUTPUT_ROOT="${1:-results/high_level_fdm_selfproof}"
DATASET_DIR="${OUTPUT_ROOT}/dataset"
MODEL_DIR="${OUTPUT_ROOT}/model"
EVAL_DIR="${OUTPUT_ROOT}/eval"
ROLLOUT_PATH="${OUTPUT_ROOT}/rollout_smoke.json"
SUMMARY_PATH="${OUTPUT_ROOT}/selfproof_summary.json"

PY="${PYTHON:-/usr/bin/python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_ROOT}"

log() { echo "[selfproof] $*"; }

log "output root: ${OUTPUT_ROOT}"

# --- Step 1: build synthetic dataset --------------------------------------

log "step 1/4 build tiny synthetic dataset"
"${PY}" -m b2_fdm_mppi.cli high-fdm synth \
    --config configs/high_level_fdm.yaml \
    --output "${DATASET_DIR}" \
    --train-samples 24 \
    --val-samples 6 \
    --test-samples 6 \
    --base-seed 1001

# --- Step 2: train 3 epochs on CPU ----------------------------------------

log "step 2/4 train 3 epochs on CPU"
"${PY}" -m b2_fdm_mppi.cli high-fdm train \
    --dataset "${DATASET_DIR}" \
    --output "${MODEL_DIR}" \
    --config configs/high_level_fdm.yaml \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --seed 42 \
    --device cpu \
    --num-workers 0

# --- Step 3: evaluate best checkpoint -------------------------------------

log "step 3/4 evaluate best checkpoint on val+test"
mkdir -p "${EVAL_DIR}"
"${PY}" -m b2_fdm_mppi.cli high-fdm eval \
    --dataset "${DATASET_DIR}" \
    --checkpoint "${MODEL_DIR}/best_model.pt" \
    --output "${EVAL_DIR}" \
    --split val \
    --device cpu
"${PY}" -m b2_fdm_mppi.cli high-fdm eval \
    --dataset "${DATASET_DIR}" \
    --checkpoint "${MODEL_DIR}/best_model.pt" \
    --output "${EVAL_DIR}" \
    --split test \
    --device cpu

# --- Step 4: rollout + MPPI cost contract ---------------------------------

log "step 4/4 rollout + MPPI cost contract"
"${PY}" -m b2_fdm_mppi.cli high-fdm rollout-smoke \
    --dataset "${DATASET_DIR}" \
    --checkpoint "${MODEL_DIR}/best_model.pt" \
    --output "${ROLLOUT_PATH}" \
    --device cpu \
    --num-samples 8

# --- Aggregate verdicts ---------------------------------------------------

log "aggregating selfproof summary"
"${PY}" - <<PY
import json
from pathlib import Path

output_root = Path("${OUTPUT_ROOT}")
model_dir = Path("${MODEL_DIR}")
eval_dir = Path("${EVAL_DIR}")
rollout_path = Path("${ROLLOUT_PATH}")
summary_path = Path("${SUMMARY_PATH}")

final_metrics_path = model_dir / "final_metrics.json"
val_eval_path = eval_dir / "val_eval_metrics.json"
test_eval_path = eval_dir / "test_eval_metrics.json"

def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

final = _load(final_metrics_path)
val_eval = _load(val_eval_path)
test_eval = _load(test_eval_path)
rollout = _load(rollout_path)

val_metrics = final["val_metrics"]
test_metrics = final["test_metrics"]

def check_beats_baseline(split, metrics):
    name = f"{split}_beats_zero_residual_baseline"
    passed = metrics["pose_xy_abs_err"] <= metrics["pose_zero_residual_baseline"] + 1e-6
    return name, passed, {
        "pose_xy_abs_err": float(metrics["pose_xy_abs_err"]),
        "pose_zero_residual_baseline": float(metrics["pose_zero_residual_baseline"]),
    }

checks = {}
for name, passed, detail in [
    check_beats_baseline("val", val_metrics),
    check_beats_baseline("test", test_metrics),
]:
    checks[name] = {"passed": bool(passed), "detail": detail}

risk_valid = 0.0 <= float(val_metrics["risk_brier"]) <= 1.0
checks["val_risk_brier_in_unit_interval"] = {
    "passed": bool(risk_valid),
    "detail": {"risk_brier": float(val_metrics["risk_brier"])},
}

rollout_shape_ok = (
    rollout["pose_param_shape"][-1] == 4
    and rollout["pose_delta_shape"][-1] == 3
    and len(rollout["risk_prob_shape"]) == 3
    and rollout["cost_shape"] == [rollout["num_samples"]]
)
checks["rollout_shapes_match_mppi_contract"] = {
    "passed": bool(rollout_shape_ok and rollout["finite"]),
    "detail": {
        "pose_param_shape": rollout["pose_param_shape"],
        "pose_delta_shape": rollout["pose_delta_shape"],
        "risk_prob_shape": rollout["risk_prob_shape"],
        "cost_shape": rollout["cost_shape"],
        "finite": rollout["finite"],
    },
}

checks["checkpoint_artifacts_exist"] = {
    "passed": all(
        (model_dir / name).exists()
        for name in ("best_model.pt", "model.pt", "schema.json", "final_metrics.json")
    ),
    "detail": [str(model_dir / name) for name in ("best_model.pt", "model.pt", "schema.json", "final_metrics.json")],
}

overall_passed = all(entry["passed"] for entry in checks.values())
summary = {
    "output_root": str(output_root),
    "overall_passed": bool(overall_passed),
    "checks": checks,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
    "val_eval_metrics": val_eval.get("metrics", {}),
    "test_eval_metrics": test_eval.get("metrics", {}),
    "rollout_smoke": rollout,
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
if not overall_passed:
    raise SystemExit(1)
PY

log "selfproof summary written to ${SUMMARY_PATH}"
