"""Helpers for creating simulation result directories."""

from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path
from typing import Any


def create_results_path(results_config: dict[str, Any] | str) -> Path:
    if isinstance(results_config, str):
        root = Path(results_config)
        run_name = None
        overwrite = False
    else:
        root = Path(str(results_config["root"]))
        run_name = results_config.get("run_name")
        overwrite = bool(results_config.get("overwrite", False))

    path = root / str(run_name) if run_name else root / _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if overwrite and path.exists():
        _remove_run_directory(path, root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remove_run_directory(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to overwrite unsafe results path: {path}")
    shutil.rmtree(resolved_path)
