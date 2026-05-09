"""Dataset I/O for the High-Level FDM.

One dataset directory contains:

    manifest.json       schema + generator config + per-split sizes
    train.npz           stacked arrays for the training split
    val.npz             stacked arrays for the validation split
    test.npz            stacked arrays for the test split

Each `*.npz` contains the same keys as one `HighLevelFdmSample`, but with
a leading batch dimension:

    state_history     (N, H, STATE_DIM)
    map_patch         (N, C, Gh, Gw)
    control_sequence  (N, N_h, CONTROL_DIM)
    pose_target       (N, N_h, POSE_PARAM_DIM)
    risk_target       (N, N_h, K)
    risk_mask         (N, N_h, K)
    seeds             (N,) int64

This module loads those files. The builder that writes them lives in
`synthetic_dataset_builder.py` to keep numpy-heavy code isolated. The
dataset class itself requires torch because it is a `Dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from b2_fdm_mppi.high_level_fdm.schema import HighLevelFdmSchema


SPLITS = ("train", "val", "test")
REQUIRED_ARRAYS = (
    "state_history",
    "map_patch",
    "control_sequence",
    "pose_target",
    "risk_target",
    "risk_mask",
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset_dir"] = str(dataset_dir)
    return manifest


def write_manifest(
    dataset_dir: Path,
    *,
    schema: HighLevelFdmSchema,
    dt: float,
    split_sizes: dict[str, int],
    generator_config: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": schema.to_dict(),
        "dt": float(dt),
        "split_sizes": {str(k): int(v) for k, v in split_sizes.items()},
        "generator_config": generator_config or {},
    }
    if extras:
        manifest["extras"] = extras
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HighLevelFdmDataset(Dataset):
    """Loads one split of a High-Level FDM dataset into memory."""

    def __init__(self, dataset_dir: str | Path, *, split: str = "train") -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        with np.load(self.dataset_dir / f"{split}.npz") as data:
            for key in REQUIRED_ARRAYS:
                if key not in data.files:
                    raise ValueError(f"split {split} missing array {key}")
            self.state_history = np.asarray(data["state_history"], dtype=np.float32)
            self.map_patch = np.asarray(data["map_patch"], dtype=np.float32)
            self.control_sequence = np.asarray(data["control_sequence"], dtype=np.float32)
            self.pose_target = np.asarray(data["pose_target"], dtype=np.float32)
            self.risk_target = np.asarray(data["risk_target"], dtype=np.float32)
            self.risk_mask = np.asarray(data["risk_mask"], dtype=np.float32)
            self.seeds = (
                np.asarray(data["seeds"], dtype=np.int64) if "seeds" in data.files else None
            )

        n = int(self.state_history.shape[0])
        for arr_name in REQUIRED_ARRAYS:
            arr = getattr(self, arr_name)
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"inconsistent split size for {arr_name}: "
                    f"{arr.shape[0]} != {n}"
                )

    def __len__(self) -> int:
        return int(self.state_history.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "state_history": torch.from_numpy(self.state_history[index]),
            "map_patch": torch.from_numpy(self.map_patch[index]),
            "control_sequence": torch.from_numpy(self.control_sequence[index]),
            "pose_target": torch.from_numpy(self.pose_target[index]),
            "risk_target": torch.from_numpy(self.risk_target[index]),
            "risk_mask": torch.from_numpy(self.risk_mask[index]),
        }
