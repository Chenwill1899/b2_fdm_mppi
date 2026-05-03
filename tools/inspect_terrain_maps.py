#!/usr/bin/env python3
"""Deprecated wrapper for archived terrain-map inspection."""

from __future__ import annotations

import sys

from archive.tools import inspect_terrain_maps as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
_main = _impl.main


def main() -> None:
    print("DEPRECATED: Stage-specific scripts now live under `archive/tools/`.", file=sys.stderr)
    _main()


if __name__ == "__main__":
    main()
