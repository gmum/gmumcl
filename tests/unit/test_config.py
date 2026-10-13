"""Unit tests for config loading, overrides and CLI parsing."""

import sys

from gmumcl.util.config import apply_overrides, parse_args


def test_apply_overrides_creates_and_replaces_nested_keys():
    config = {"method": {"lambda": 1.0}}
    apply_overrides(config, ["method.lambda=0.5", "figures.regions=True", "name=abc"])
    assert config["method"]["lambda"] == 0.5
    assert config["figures"]["regions"] is True
    assert config["name"] == "abc"


def test_repeated_overrides_are_kept(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--config", "c.yaml", "-O", "method.lambda=5", "-O", "seed=1"]
    )
    args = parse_args()
    assert args.overrides == ["method.lambda=5", "seed=1"]
