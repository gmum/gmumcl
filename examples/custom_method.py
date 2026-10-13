"""Define and run a custom plugin: a cosine-similarity anchor.

Run: python examples/custom_method.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch.nn.functional as F

from gmumcl import train
from gmumcl.cl.baselines import REGISTRY
from gmumcl.cl.growth import aligned_rows
from gmumcl.cl.plugin import MethodPluginABC


class CosineAnchor(MethodPluginABC):
    """lambda * sum(1 - cos(theta, theta*)) towards the task-end snapshot."""

    def on_task_end(self, task_id, task) -> None:
        self.anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

    def forward(self, x, y, loss, logits):
        if not hasattr(self, "anchor"):
            return loss, logits
        penalty = sum(
            (
                1.0
                - F.cosine_similarity(
                    aligned_rows(parameter, self.anchor[name]).flatten(),
                    self.anchor[name].flatten(),
                    dim=0,
                )
            ).sum()
            for name, parameter in self.model.named_parameters()
        )
        return loss + float(self.config.get("lambda", 1.0)) * penalty, logits


REGISTRY["cosine_anchor"] = CosineAnchor

CONFIG = {
    "seed": 0,
    "data": {
        "name": "toy",
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
    "method": {"name": "cosine_anchor", "lambda": 0.1},
}


def main() -> None:
    metrics = train.run(CONFIG)
    print("average incremental accuracy: %.3f" % metrics["average_incremental_accuracy"])


if __name__ == "__main__":
    main()
