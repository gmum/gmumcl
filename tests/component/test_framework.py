"""Component tests for the cl framework: splits, trainer, metrics."""


from gmumcl.cl.metrics import AccuracyMatrix
from gmumcl.cl.models import MLP
from gmumcl.cl.scenarios import synthetic_tasks
from gmumcl.cl.trainer import Trainer, TrainerConfig


def test_splits_are_disjoint_and_cover_each_class():
    tasks = synthetic_tasks(num_classes=4, classes_per_task=2, dimension=4, samples_per_class=40)
    assert len(tasks) == 2
    for task in tasks:
        fitting = len(task.fit_loader.dataset)
        calibration = len(task.calibration_loader.dataset)
        assert fitting > 0 and calibration > 0
        targets = [int(task.fit_loader.dataset.dataset.targets[i]) for i in task.fit_loader.dataset.indices]
        assert set(targets) <= set(task.classes)


def test_trainer_runs_without_plugins_and_fills_matrix():
    tasks = synthetic_tasks(num_classes=4, classes_per_task=2, dimension=4, samples_per_class=40)
    model = MLP(input_dim=4, n_classes=4, hidden=16)
    trainer = Trainer(model, [], TrainerConfig(epochs=1, lr=0.01))
    matrix = trainer.run(tasks)
    assert len(matrix.rows) == 2
    assert all(0.0 <= value <= 1.0 for row in matrix.rows for value in row)
    assert matrix.final_old_new()[1] >= 0.0


def test_accuracy_matrix_metrics():
    matrix = AccuracyMatrix(rows=[[0.9], [0.6, 0.8], [0.5, 0.7, 0.9]])
    assert abs(matrix.old_accuracy(2) - 0.6) < 1e-9
    assert abs(matrix.new_accuracy(2) - 0.9) < 1e-9
    assert abs(matrix.forgetting() - (0.4 + 0.1) / 2) < 1e-9
    assert 0.0 <= matrix.average_incremental_accuracy() <= 1.0
