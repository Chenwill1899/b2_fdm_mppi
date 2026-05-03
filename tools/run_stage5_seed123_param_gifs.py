#!/usr/bin/env python3
"""Deprecated wrapper for archived Stage 5 fixed-seed GIF generation."""

from __future__ import annotations

import sys

from archive.tools import run_stage5_seed123_param_gifs as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
_main = _impl.main


def main() -> None:
    print("DEPRECATED: Stage-specific scripts now live under `archive/tools/`.", file=sys.stderr)
    _main()


if __name__ == "__main__":
    main()
