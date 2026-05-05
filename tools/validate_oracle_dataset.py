#!/usr/bin/env python3
"""Validate oracle dataset splits and create a summary figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from b2_fdm_mppi.data.oracle_dataset_validator import validate_oracle_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    print("DEPRECATED: use `python3 tools/fdm_mppi.py dataset validate ...`.", file=sys.stderr)
    quality = validate_oracle_dataset(Path(args.dataset), Path(args.output or args.dataset))
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
