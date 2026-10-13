"""Config and seeding helpers.

Base config: a YAML file, optionally wrapped in a single top-level key.
Overrides: argparse ``-O key.path=value``, parsed with ast.literal_eval.
"""

from __future__ import annotations

import argparse
import ast
import random
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, unwrapping a single top-level wrapper key if present."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if isinstance(config, dict) and len(config) == 1:
        (only,) = config.values()
        if isinstance(only, dict):
            return only
    return config


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> None:
    """Apply dot-notation overrides in place, e.g. ``method.lambda=0.5``."""
    for item in overrides:
        key, value = item.split("=", 1)
        parts = key.split(".")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            node[parts[-1]] = value


def parse_args() -> argparse.Namespace:
    """Canonical CLI: --config, optional --output_dir and -O overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("-O", "--overrides", action="append", default=[])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed random, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
