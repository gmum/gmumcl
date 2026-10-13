"""Unit tests for the trainer: optimizer choice and per-task iteration cap."""

import torch

from gmumcl.cl.models import MLP
from gmumcl.cl.plugin import MethodPluginABC
from gmumcl.cl.scenarios import synthetic_tasks
from gmumcl.cl.trainer import Trainer, TrainerConfig


class _StepCounter(MethodPluginABC):
    def __init__(self):
        super().__init__()
        self.steps = 0

    def on_after_backward(self):
        self.steps += 1


def _tasks():
    return synthetic_tasks(
        num_classes=4,
        classes_per_task=2,
        dimension=6,
        samples_per_class=6,
        min_calibration=4,
        batch_size=4,
        seed=0,
    )


def test_iterations_cap_counts_exact_optimizer_steps():
    model = MLP(input_dim=6, n_classes=4, hidden=8)
    counter = _StepCounter()
    trainer = Trainer(model, [counter], TrainerConfig(epochs=99, iterations=5, lr=0.01))
    trainer.run(_tasks())
    assert counter.steps == 10  # 2 tasks x 5 steps


def test_optimizer_selection():
    model = MLP(input_dim=6, n_classes=4, hidden=8)
    trainer = Trainer(model, [], TrainerConfig(optimizer="sgd", momentum=0.9, lr=0.05))
    optimizer = trainer._make_optimizer()
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9
    adam = Trainer(model, [], TrainerConfig())._make_optimizer()
    assert isinstance(adam, torch.optim.Adam)
