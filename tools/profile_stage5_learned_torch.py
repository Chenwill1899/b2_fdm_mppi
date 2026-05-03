#!/usr/bin/env python3
"""Deprecated wrapper for archived Stage 5 runtime profiling."""

from __future__ import annotations

import sys

from archive.tools import profile_stage5_learned_torch as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
_main = _impl.main


def main() -> None:
    print("DEPRECATED: Stage-specific scripts now live under `archive/tools/`.", file=sys.stderr)
    _main()


if __name__ == "__main__":
    main()
