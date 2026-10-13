"""Quickstart: a tiny class-incremental run on CPU, no downloads.

Run: python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmumcl import train

CONFIG = {
    "seed": 0,
    "data": {
        "name": "toy",
        "scenario": "class_incremental",
        "dimension": 6,
        "num_classes": 4,
        "classes_per_task": 2,
        "samples_per_class": 40,
        "calibration_fraction": 0.1,
        "min_calibration": 20,
        "batch_size": 32,
    },
    "model": {"name": "mlp", "hidden": 16, "layers": 2, "growing_head": True},
    "trainer": {"epochs": 1, "lr": 0.01, "device": "cpu"},
    "method": {"name": "ewc", "lambda": 1.0},
}


def main() -> None:
    metrics = train.run(CONFIG)
    for index, row in enumerate(metrics["accuracy_matrix"]):
        print("after task %d: %s" % (index + 1, ["%.3f" % value for value in row]))
    print("average incremental accuracy: %.3f" % metrics["average_incremental_accuracy"])
    print("forgetting: %.3f" % metrics["forgetting"])


if __name__ == "__main__":
    main()
