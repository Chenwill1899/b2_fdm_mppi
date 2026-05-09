"""Build a train/val/test dataset of High-Level FDM samples.

This module is numpy-only; it does not import torch. The produced directory
is consumed by `HighLevelFdmDataset` in `dataset.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from b2_fdm_mppi.high_level_fdm.dataset import REQUIRED_ARRAYS, write_manifest
from b2_fdm_mppi.high_level_fdm.schema import HighLevelFdmSchema
from b2_fdm_mppi.high_level_fdm.synthetic import (
    HighLevelFdmSyntheticGenerator,
    SyntheticConfig,
    synthetic_config_to_dict,
)


@dataclass
class DatasetBuildSpec:
    train_samples: int = 128
    val_samples: int = 32
    test_samples: int = 32
    base_seed: int = 123


def build_synthetic_dataset(
    *,
    output_dir: str | Path,
    generator: HighLevelFdmSyntheticGenerator | None = None,
    spec: DatasetBuildSpec | None = None,
    synthetic_config: SyntheticConfig | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = generator or HighLevelFdmSyntheticGenerator(synthetic_config or SyntheticConfig())
    spec = spec or DatasetBuildSpec()

    schema = generator.schema()
    split_sizes = {
        "train": int(spec.train_samples),
        "val": int(spec.val_samples),
        "test": int(spec.test_samples),
    }

    # Assign distinct, deterministic seed ranges per split so a larger
    # dataset can be rebuilt without regenerating old samples.
    seeds = _split_seeds(spec.base_seed, split_sizes)
    written: dict[str, dict[str, Any]] = {}
    for split, split_seeds in seeds.items():
        arrays = _generate_split_arrays(generator, split_seeds)
        path = output_dir / f"{split}.npz"
        np.savez_compressed(str(path), **arrays, seeds=np.asarray(split_seeds, dtype=np.int64))
        written[split] = {
            "path": str(path),
            "num_samples": int(split_sizes[split]),
        }

    generator_config_dict = synthetic_config_to_dict(generator.config)
    manifest = write_manifest(
        output_dir,
        schema=schema,
        dt=float(generator.config.dt),
        split_sizes=split_sizes,
        generator_config=generator_config_dict,
        extras={"build_spec": {
            "train_samples": int(spec.train_samples),
            "val_samples": int(spec.val_samples),
            "test_samples": int(spec.test_samples),
            "base_seed": int(spec.base_seed),
        }},
    )
    summary = {
        "output_dir": str(output_dir),
        "splits": written,
        "schema": manifest["schema"],
        "generator_config": generator_config_dict,
        "total_samples": int(sum(split_sizes.values())),
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _split_seeds(base_seed: int, sizes: dict[str, int]) -> dict[str, list[int]]:
    # Reserve non-overlapping seed ranges per split so rebuilding a larger
    # dataset stays deterministic.
    offsets = {"train": 0, "val": 10**7, "test": 2 * 10**7}
    return {
        split: [int(base_seed) + offsets[split] + idx for idx in range(size)]
        for split, size in sizes.items()
    }


def _generate_split_arrays(
    generator: HighLevelFdmSyntheticGenerator,
    seeds: Iterable[int],
) -> dict[str, np.ndarray]:
    state_history: list[np.ndarray] = []
    map_patch: list[np.ndarray] = []
    control_sequence: list[np.ndarray] = []
    pose_target: list[np.ndarray] = []
    risk_target: list[np.ndarray] = []
    risk_mask: list[np.ndarray] = []

    for seed in seeds:
        sample = generator.generate_sample(int(seed))
        state_history.append(sample.state_history)
        map_patch.append(sample.map_patch)
        control_sequence.append(sample.control_sequence)
        pose_target.append(sample.pose_target)
        risk_target.append(sample.risk_target)
        risk_mask.append(sample.risk_mask)

    def _stack(arrays: list[np.ndarray]) -> np.ndarray:
        if not arrays:
            raise ValueError("cannot stack an empty sample list")
        return np.stack(arrays, axis=0).astype(np.float32, copy=False)

    arrays = {
        "state_history": _stack(state_history),
        "map_patch": _stack(map_patch),
        "control_sequence": _stack(control_sequence),
        "pose_target": _stack(pose_target),
        "risk_target": _stack(risk_target),
        "risk_mask": _stack(risk_mask),
    }
    for key in REQUIRED_ARRAYS:
        if key not in arrays:
            raise RuntimeError(f"dataset builder missing array {key}")
    return arrays


__all__ = [
    "DatasetBuildSpec",
    "build_synthetic_dataset",
]
