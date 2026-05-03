#!/usr/bin/env python3
"""Build train/val/test npz splits from collected oracle episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from b2_fdm_mppi.data.oracle_dataset import build_oracle_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)

    print("DEPRECATED: use `python3 tools/fdm_mppi.py dataset build ...`.", file=sys.stderr)
    summary = build_oracle_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
