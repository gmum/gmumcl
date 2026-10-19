"""Smoke tests: tiny runs on CPU in well under a minute."""

import math

from gmumcl import train


def _toy(**overrides):
    config = {
        "seed": 0,
        "data": {
            "name": "toy",
            "dimension": 6,
            "num_classes": 4,
            "classes_per_task": 2,
            "samples_per_class": 40,
            "calibration_fraction": 0.1,
            "min_calibration": 30,
            "batch_size": 32,
        },
        "model": {"name": "mlp", "hidden": 16, "layers": 2},
        "trainer": {"epochs": 1, "lr": 0.01, "device": "cpu"},
        "method": {"name": "naive"},
    }
    config.update(overrides)
    return config


def test_toy_class_incremental_run():
    metrics = train.run(_toy())
    assert metrics["scenario"] == "class_incremental"
    assert len(metrics["accuracy_matrix"]) == 2
    assert len(metrics["old_class_matrix"]) == 2
    assert len(metrics["task_aware_matrix"]) == 2
    assert len(metrics["new_class_margin"]) == 2
    assert metrics["validation_matrix"] == []
    assert math.isfinite(metrics["average_incremental_accuracy"])
    assert math.isfinite(metrics["final_average_accuracy"])
    assert math.isfinite(metrics["final_old_class_accuracy"])
    assert math.isfinite(metrics["forgetting"])


def test_toy_task_incremental_run_uses_the_oracle():
    config = _toy()
    config["data"] = {**config["data"], "scenario": "task_incremental"}
    metrics = train.run(config)
    assert metrics["scenario"] == "task_incremental"
    assert len(metrics["accuracy_matrix"]) == 2


def test_toy_permuted_run():
    config = _toy()
    config["data"] = {**config["data"], "scenario": "permuted", "num_tasks": 2}
    metrics = train.run(config)
    assert metrics["scenario"] == "permuted"
    assert len(metrics["accuracy_matrix"]) == 2
