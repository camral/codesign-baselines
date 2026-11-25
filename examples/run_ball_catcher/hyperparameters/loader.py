"""Shared loader for the per-method hyperparameter YAML files in this folder.

Each ``run_*.py`` script in ``run_ball_catcher`` has a matching YAML file here
(e.g. ``run_fasttd3_rl.py`` <-> ``fasttd3_rl.yaml``) that defines the method's
tunable hyperparameters, defaulted to the values used by that method's
original implementation elsewhere in this repository. Scripts load their
file with :func:`load_hyperparameters` and feed the values in as argparse
defaults, so the CLI remains the single source of truth for what can be
overridden while the YAML file remains the single source of truth for what
the default values are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

HYPERPARAMETERS_DIR = Path(__file__).resolve().parent


def load_hyperparameters(name: str) -> Dict[str, Any]:
    """Load ``<name>.yaml`` from this folder as a flat dict of defaults."""

    path = HYPERPARAMETERS_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Hyperparameter file {path} must contain a YAML mapping")
    return config
