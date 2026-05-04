#!/usr/bin/env python3
"""CLI for training Sequence FDM V2 with curriculum learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from b2_fdm_mppi.data.sequence_fdm_collector import build_sequence_fdm_windows
from b2_fdm_mppi.training.sequence_fdm_v2 import train_sequence_fdm_v2


def main():
    parser = argparse.ArgumentParser(description="Train Sequence FDM V2")
    parser.add_argument("--data-dir", required=True, help="Directory containing collected episode .npz files")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20], help="Curriculum horizons")
    parser.add_argument("--epochs", type=int, nargs="+", default=[50, 50, 100], help="Epochs per phase")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 5e-4, 1e-4], help="Learning rates per phase")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--w-risk", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not (len(args.horizons) == len(args.epochs) == len(args.lrs)):
        raise ValueError("--horizons, --epochs, and --lrs must have the same length")

    data_dir = Path(args.data_dir)
    episode_files = sorted(data_dir.glob("episode_*.npz"))
    if not episode_files:
        raise ValueError(f"No episode files found in {data_dir}")

    print(f"Loading {len(episode_files)} episodes...")
    all_windows = []
    for ep_path in episode_files:
        windows = build_sequence_fdm_windows(ep_path, horizon_steps=max(args.horizons), stride=1)
        all_windows.extend(windows)
    print(f"Extracted {len(all_windows)} training windows")

    curriculum = list(zip(args.horizons, args.epochs, args.lrs))
    metrics = train_sequence_fdm_v2(
        windows=all_windows,
        output_dir=args.output_dir,
        hidden_dims=args.hidden_dims,
        curriculum_phases=curriculum,
        batch_size=args.batch_size,
        w_risk=args.w_risk,
        patience=args.patience,
        device=args.device,
    )

    with open(Path(args.output_dir) / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Best val loss: {metrics['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
