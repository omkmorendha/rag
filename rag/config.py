"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config file, returning an empty config when it is blank."""
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    """Return one stage config block."""
    value = config.get(stage, {})
    if not isinstance(value, dict):
        raise ValueError(f"{stage} config must be a mapping")
    return value
