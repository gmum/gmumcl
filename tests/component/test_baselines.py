"""Component tests: every baseline runs and produces finite losses."""

import math

import pytest
import torch

from gmumcl.cl.baselines import build as build_baseline
from gmumcl.cl.models import MLP
from gmumcl.cl.scenarios import synthetic_tasks
from gmumcl.cl.trainer import Trainer, TrainerConfig

METHODS = [
    "naive",
    "l2",
    "ewc",
    "ewc_dr",
    "online_ewc",
    "si",
    "mas",
    "lwf",
    "kfac",
    "ncl",
    "owm",
    "gpm",
    "intact",
    "sharpening",
]


@pytest.mark.parametrize("name", METHODS)
def test_baseline_runs_two_tasks(name):
    tasks = synthetic_tasks(
        num_classes=4,
        classes_per_task=2,
        dimension=6,
        samples_per_class=60,
        min_calibration=20,
        batch_size=32,
        seed=1,
    )
    model = MLP(input_dim=6, n_classes=4, hidden=16)
    plugin = build_baseline(name, {"lambda": 0.1, "kfac_samples": 16, "threshold": 0.05})
    trainer = Trainer(model, [plugin], TrainerConfig(epochs=1, lr=0.01))
    matrix = trainer.run(tasks[:2])
    assert len(matrix.rows) == 2
    assert all(math.isfinite(value) for row in matrix.rows for value in row)


def _kfac_after_first_task(**settings):
    tasks = synthetic_tasks(
        num_classes=4,
        classes_per_task=2,
        dimension=6,
        samples_per_class=60,
        min_calibration=20,
        batch_size=32,
        seed=1,
    )
    model = MLP(input_dim=6, n_classes=4, hidden=16)
    plugin = build_baseline("kfac", {"lambda": 2.0, "kfac_samples": 16, **settings})
    trainer = Trainer(model, [plugin], TrainerConfig(epochs=1, lr=0.01))
    trainer.run(tasks[:1])
    return model, plugin


def test_kfac_penalises_updates_from_the_task_end_snapshot():
    model, plugin = _kfac_after_first_task()
    loss = torch.ones(())
    x = torch.zeros(2, 6)
    y = torch.zeros(2, dtype=torch.long)
    composed, _ = plugin.forward(x, y, loss, model(x))
    assert torch.isclose(composed, loss), "penalty must vanish at the stored parameters"
    with torch.no_grad():
        model.head.weight[0].add_(0.5)
    composed, _ = plugin.forward(x, y, loss, model(x))
    assert composed > loss


def test_kfac_two_weight_runs():
    model, plugin = _kfac_after_first_task(
        two_weight=True, lambda_center=1.0, lambda_shape=1.0
    )
    with torch.no_grad():
        model.features[0].weight.add_(0.1)
    x = torch.zeros(2, 6)
    composed, _ = plugin.forward(x, torch.zeros(2, dtype=torch.long), torch.ones(()), model(x))
    assert torch.isfinite(composed) and composed > 0.0
