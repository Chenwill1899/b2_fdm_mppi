#!/usr/bin/env python3
"""Deprecated wrapper for oracle episode collection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2_fdm_mppi.data import collect_oracle_episode as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
_main = _impl.main


def collect_oracle_episode(*args, **kwargs):
    _impl.load_config = globals()["load_config"]
    _impl.OmniMppiSimulationRunner = globals()["OmniMppiSimulationRunner"]
    _impl.create_omni_controller = globals()["create_omni_controller"]
    _impl.build_episode_npz = globals()["build_episode_npz"]
    return _impl.collect_oracle_episode(*args, **kwargs)


def main() -> None:
    print("DEPRECATED: use `python3 tools/fdm_mppi.py dataset collect ...` for multi-episode collection.", file=sys.stderr)
    _main()


if __name__ == "__main__":
    main()
