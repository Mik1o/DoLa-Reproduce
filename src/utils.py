"""Shared utility helpers for scripts and future experiment code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return data


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create an output directory if it does not already exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_root() -> Path:
    """Return the repository root based on the current file location."""
    return Path(__file__).resolve().parent.parent
