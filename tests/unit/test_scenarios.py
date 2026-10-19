"""Unit tests for the five scenario builders."""

import torch

from gmumcl.cl.scenarios import (
    class_incremental_tasks,
    domain_incremental_tasks,
    instance_incremental_tasks,
    permuted_tasks,
    shift_transform,
    synthetic_pair,
    task_incremental_tasks,
)


def _pair(num_classes: int = 4, dimension: int = 4, samples_per_class: int = 40):
    return synthetic_pair(
        num_classes=num_classes, dimension=dimension, samples_per_class=samples_per_class
    )


def _labels(task, split: str = "fit"):
    subset = getattr(task, split + "_loader").dataset
    base = subset.dataset
    targets = getattr(base, "targets")
    return [int(targets[i]) for i in subset.indices]


def _indices(task, split: str = "fit"):
    return set(getattr(task, split + "_loader").dataset.indices)


def _split_kwargs(**overrides):
    settings = {"calibration_fraction": 0.1, "min_calibration": 20, "batch_size": 16, "seed": 0}
    settings.update(overrides)
    return settings


def test_class_incremental_splits_are_disjoint_and_cover_each_class():
    train, test = _pair()
    tasks = class_incremental_tasks(
        train, test, [[0, 1], [2, 3]], **_split_kwargs()
    )
    assert [task.classes for task in tasks] == [[0, 1], [2, 3]]
    for task in tasks:
        assert set(_labels(task, "fit")) <= set(task.classes)
        assert len(task.fit_loader.dataset) > 0
        assert len(task.calibration_loader.dataset) > 0
        assert set(_labels(task, "test")) <= set(task.classes)


def test_class_incremental_validation_split_is_disjoint():
    train, test = _pair()
    tasks = class_incremental_tasks(
        train, test, [[0, 1], [2, 3]], **_split_kwargs(validation_fraction=0.2)
    )
    for task in tasks:
        assert task.validation_loader is not None
        fitting = _indices(task, "fit")
        calibration = _indices(task, "calibration")
        validation = _indices(task, "validation")
        assert fitting and calibration and validation
        assert not fitting & calibration
        assert not fitting & validation
        assert not calibration & validation


def test_class_incremental_calibration_in_fit_trains_on_the_calibration_images():
    train, test = _pair()
    tasks = class_incremental_tasks(
        train,
        test,
        [[0, 1], [2, 3]],
        **_split_kwargs(calibration_in_fit=True, validation_fraction=0.1),
    )
    for task in tasks:
        fitting = _indices(task, "fit")
        calibration = _indices(task, "calibration")
        validation = _indices(task, "validation")
        assert calibration <= fitting
        assert not validation & fitting


def test_task_incremental_split_matches_class_incremental():
    train, test = _pair()
    kwargs = _split_kwargs()
    class_tasks = class_incremental_tasks(train, test, [[0, 1], [2, 3]], **kwargs)
    task_tasks = task_incremental_tasks(train, test, [[0, 1], [2, 3]], **kwargs)
    assert [task.classes for task in task_tasks] == [task.classes for task in class_tasks]
    for task, reference in zip(task_tasks, class_tasks):
        assert _labels(task, "test") == _labels(reference, "test")


def test_instance_incremental_partitions_every_instance_once():
    train, test = _pair()
    tasks = instance_incremental_tasks(
        train, test, num_tasks=2, **_split_kwargs()
    )
    assert len(tasks) == 2
    all_labels = []
    for task in tasks:
        labels = _labels(task, "fit") + _labels(task, "calibration")
        assert set(labels) == {0, 1, 2, 3}, "every task mixes all classes"
        all_labels.extend(labels)
    assert sorted(all_labels) == sorted([int(y) for y in train.targets])


def test_domain_incremental_keeps_classes_and_changes_inputs():
    train, test = _pair()
    shift = torch.ones(4)
    tasks = domain_incremental_tasks(
        train,
        test,
        [shift_transform(torch.zeros(4)), shift_transform(shift)],
        **_split_kwargs(),
    )
    assert [task.classes for task in tasks] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    first, second = tasks
    x_first, _ = first.fit_loader.dataset.dataset[0]
    x_second, _ = second.fit_loader.dataset.dataset[0]
    assert torch.allclose(x_second, x_first + shift)


def test_permuted_tasks_identity_first_then_permutations():
    train, test = _pair()
    tasks = permuted_tasks(
        train, test, num_tasks=3, **_split_kwargs()
    )
    assert [task.classes for task in tasks] == [[0, 1, 2, 3]] * 3
    reference, _ = train[0]
    first, _ = tasks[0].fit_loader.dataset.dataset[0]
    assert torch.equal(first, reference), "task 0 keeps the identity permutation"
    changed = any(
        not torch.equal(tasks[index].fit_loader.dataset.dataset[0][0], reference)
        for index in (1, 2)
    )
    assert changed
