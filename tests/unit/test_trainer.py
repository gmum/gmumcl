"""Unit tests for the trainer: optimizer choice, hooks and per-task caps."""

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


class _TaskRecorder(MethodPluginABC):
    def __init__(self):
        super().__init__()
        self.seen = []

    def setup_task(self, task_id, task=None):
        self.seen.append((task_id, task))


class _FixedScores(MethodPluginABC):
    """Predicts class 0 for every input, whatever the model says."""

    def scores(self, x):
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(x.shape[0], 1)


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


def test_setup_task_receives_the_task_data():
    tasks = _tasks()
    recorder = _TaskRecorder()
    model = MLP(input_dim=6, n_classes=4, hidden=8)
    Trainer(model, [recorder], TrainerConfig(epochs=1, lr=0.01)).run(tasks)
    assert [task_id for task_id, _ in recorder.seen] == [0, 1]
    assert [task for _, task in recorder.seen] == tasks


def test_trainer_evaluates_through_the_plugin_scores():
    tasks = _tasks()
    model = MLP(input_dim=6, n_classes=4, hidden=8)
    trainer = Trainer(model, [_FixedScores()], TrainerConfig(epochs=1, lr=0.01))
    matrix = trainer.run(tasks)
    assert matrix.rows[-1][0] == 0.5, "only the class-0 samples are predicted correctly"
    assert matrix.rows[-1][1] == 0.0, "no task-1 sample is ever predicted correctly"


def test_trainer_fills_the_extended_matrices():
    tasks = _tasks()
    model = MLP(input_dim=6, n_classes=4, hidden=8)
    matrix = Trainer(model, [], TrainerConfig(epochs=1, lr=0.01)).run(tasks)
    assert len(matrix.old_class_rows) == 2
    assert len(matrix.task_aware_rows) == 2
    assert len(matrix.new_class_margin_rows) == 2
    assert matrix.new_class_margin_rows[0] == [None]
    assert matrix.new_class_margin_rows[1][0] is not None


def test_trainer_records_validation_rows_when_the_task_has_a_loader():
    tasks = synthetic_tasks(
        num_classes=4,
        classes_per_task=2,
        dimension=6,
        samples_per_class=40,
        min_calibration=4,
        validation_fraction=0.2,
        batch_size=4,
        seed=0,
    )
    model = MLP(input_dim=6, n_classes=4, hidden=8)
    matrix = Trainer(model, [], TrainerConfig(epochs=1, lr=0.01)).run(tasks)
    assert len(matrix.validation_rows) == 2
    assert all(value is not None for value in matrix.validation_rows[-1])
    assert matrix.validation_final_average_accuracy() is not None
