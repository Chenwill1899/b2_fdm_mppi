#!/usr/bin/env python3
"""Deprecated wrapper for `python3 tools/fdm_mppi.py run`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.simulation import run_omni_mppi as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
_main = _impl.main


def main() -> None:
    print("DEPRECATED: use `python3 tools/fdm_mppi.py run ...`.", file=sys.stderr)
    _main()


if __name__ == "__main__":
    main()
