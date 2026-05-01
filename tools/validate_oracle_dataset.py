#!/usr/bin/env python3
"""Validate oracle dataset splits and create a summary figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.data.oracle_dataset_validator import validate_oracle_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    quality = validate_oracle_dataset(Path(args.dataset), Path(args.output))
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
