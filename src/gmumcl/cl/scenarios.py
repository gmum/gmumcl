"""Task builders for the five incremental scenarios.

Every builder returns a list of :class:`TaskData`, one per incremental task.
Each task's *training* images are split into a fitting subset (the network
trains on it) and a held-out calibration subset (kept for calibration-style
methods); the split and shuffle generators are seeded from the config seed.

Scenarios:

- ``class_incremental``: disjoint class groups; evaluation over all classes
  seen so far (the trainer never masks logits);
- ``task_incremental``: the same split, evaluated with the task oracle (the
  trainer masks logits to the task's own classes);
- ``domain_incremental``: every task carries all classes under a different
  input transform (e.g. rotations), one shared head;
- ``instance_incremental``: random partitions of all instances, classes mixed
  inside every task;
- ``permuted``: every task applies a fixed random input permutation, classes
  unchanged (Permuted MNIST-style).

``split_mnist``, ``split_cifar10`` and ``split_cifar100`` are class-incremental
thin torchvision wrappers; ``rotated_mnist`` and ``permuted_mnist`` are the
canonical domain-incremental and permuted wrappers; ``synthetic_tasks`` builds
Gaussian clusters for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


@dataclass
class TaskData:
    """One incremental task: loaders and its classes (original label ids)."""

    task_id: int
    classes: List[int]
    fit_loader: DataLoader
    calibration_loader: DataLoader
    test_loader: DataLoader


def _targets(dataset: Dataset) -> Sequence[int]:
    if hasattr(dataset, "targets"):
        return list(getattr(dataset, "targets"))  # type: ignore[arg-type]
    if hasattr(dataset, "labels"):
        return list(getattr(dataset, "labels"))  # type: ignore[arg-type]
    return [int(y) for _, y in dataset]  # type: ignore[misc]


def _indices_of_classes(targets: Sequence[int], classes: Sequence[int]) -> List[int]:
    wanted = set(int(c) for c in classes)
    return [i for i, value in enumerate(targets) if int(value) in wanted]


def _split_indices(
    indices: Sequence[int],
    calibration_fraction: float,
    min_calibration: int,
    generator: torch.Generator,
):
    """Shuffle ``indices`` and split off the calibration subset."""
    if not indices:
        return [], []
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    indices = [indices[i] for i in permutation]
    calibration_size = max(min_calibration, int(calibration_fraction * len(indices)))
    calibration_size = min(calibration_size, max(len(indices) - 1, 1))
    return indices[calibration_size:], indices[:calibration_size]


def _build_task(
    task_id: int,
    dataset: Dataset,
    indices: Sequence[int],
    classes: Sequence[int],
    test_dataset: Dataset,
    test_indices: Sequence[int],
    generator: torch.Generator,
    calibration_fraction: float,
    min_calibration: int,
    batch_size: int,
    num_workers: int,
    evaluation_batch_size: int,
) -> TaskData:
    """Fit/calibration split for one task plus its evaluation loader."""
    fitting, calibration = _split_indices(
        indices, calibration_fraction, min_calibration, generator
    )
    return TaskData(
        task_id=task_id,
        classes=[int(c) for c in classes],
        fit_loader=DataLoader(
            Subset(dataset, fitting),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        calibration_loader=DataLoader(
            Subset(dataset, calibration),
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test_loader=DataLoader(
            Subset(test_dataset, test_indices),
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


def _chunks(values: Sequence[int], num_tasks: int) -> List[List[int]]:
    """Split ``values`` into ``num_tasks`` nearly equal contiguous chunks."""
    size = len(values)
    boundaries = [round(index * size / num_tasks) for index in range(num_tasks + 1)]
    return [list(values[boundaries[i] : boundaries[i + 1]]) for i in range(num_tasks)]


def class_incremental_tasks(
    train_dataset: Dataset,
    test_dataset: Dataset,
    classes_per_task: Sequence[Sequence[int]],
    calibration_fraction: float = 0.1,
    min_calibration: int = 500,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 0,
    test_batch_size: Optional[int] = None,
) -> List[TaskData]:
    """Build TaskData for every class group, with fit/calibration split."""
    train_targets = _targets(train_dataset)
    test_targets = _targets(test_dataset)
    generator = torch.Generator().manual_seed(seed)
    tasks: List[TaskData] = []
    evaluation_batch_size = test_batch_size or max(batch_size, 256)

    for task_id, classes in enumerate(classes_per_task):
        classes = [int(c) for c in classes]
        indices = _indices_of_classes(train_targets, classes)
        test_indices = _indices_of_classes(test_targets, classes)
        tasks.append(
            _build_task(
                task_id,
                train_dataset,
                indices,
                classes,
                test_dataset,
                test_indices,
                generator,
                calibration_fraction,
                min_calibration,
                batch_size,
                num_workers,
                evaluation_batch_size,
            )
        )
    return tasks


def task_incremental_tasks(
    train_dataset: Dataset,
    test_dataset: Dataset,
    classes_per_task: Sequence[Sequence[int]],
    **kwargs,
) -> List[TaskData]:
    """The class-incremental split; evaluation uses the task oracle.

    The split is identical to :func:`class_incremental_tasks`; the difference
    lives in the trainer, which masks the logits to the task's own classes
    when ``TrainerConfig.task_oracle`` is set.
    """
    return class_incremental_tasks(train_dataset, test_dataset, classes_per_task, **kwargs)


def instance_incremental_tasks(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_tasks: int = 5,
    calibration_fraction: float = 0.1,
    min_calibration: int = 500,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 0,
    test_batch_size: Optional[int] = None,
) -> List[TaskData]:
    """Random partitions of all instances; every task mixes all classes."""
    train_targets = _targets(train_dataset)
    test_targets = _targets(test_dataset)
    classes = sorted({int(value) for value in train_targets})
    generator = torch.Generator().manual_seed(seed)
    evaluation_batch_size = test_batch_size or max(batch_size, 256)
    train_order = torch.randperm(len(train_targets), generator=generator).tolist()
    test_order = torch.randperm(len(test_targets), generator=generator).tolist()

    tasks: List[TaskData] = []
    for task_id, (indices, test_indices) in enumerate(
        zip(_chunks(train_order, num_tasks), _chunks(test_order, num_tasks))
    ):
        tasks.append(
            _build_task(
                task_id,
                train_dataset,
                indices,
                classes,
                test_dataset,
                test_indices,
                generator,
                calibration_fraction,
                min_calibration,
                batch_size,
                num_workers,
                evaluation_batch_size,
            )
        )
    return tasks


class TransformedDataset(Dataset):
    """Applies ``transform`` to the input of ``dataset[i]``; labels unchanged."""

    def __init__(self, dataset: Dataset, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if isinstance(item, (tuple, list)):
            features, *rest = item
            return (self.transform(features), *rest)
        return self.transform(item)


def domain_incremental_tasks(
    train_dataset: Dataset,
    test_dataset: Dataset,
    transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    calibration_fraction: float = 0.1,
    min_calibration: int = 500,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 0,
    test_batch_size: Optional[int] = None,
) -> List[TaskData]:
    """All classes in every task; task ``i`` applies ``transforms[i]``."""
    classes = sorted({int(value) for value in _targets(train_dataset)})
    generator = torch.Generator().manual_seed(seed)
    evaluation_batch_size = test_batch_size or max(batch_size, 256)

    tasks: List[TaskData] = []
    for task_id, transform in enumerate(transforms):
        train_task = TransformedDataset(train_dataset, transform)
        test_task = TransformedDataset(test_dataset, transform)
        tasks.append(
            _build_task(
                task_id,
                train_task,
                list(range(len(train_task))),
                classes,
                test_task,
                list(range(len(test_task))),
                generator,
                calibration_fraction,
                min_calibration,
                batch_size,
                num_workers,
                evaluation_batch_size,
            )
        )
    return tasks


def permutation_transform(
    permutation: torch.Tensor, shape: Sequence[int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Flatten, reorder with ``permutation`` and restore ``shape``."""

    def apply(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1)[permutation].reshape(tuple(shape))

    return apply


def shift_transform(shift: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Translate every input by ``shift`` (a vector-domain shift)."""

    def apply(x: torch.Tensor) -> torch.Tensor:
        return x + shift.reshape(x.shape)

    return apply


def rotation_transform(angle_degrees: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Rotate image tensors by ``angle_degrees`` (torchvision functional)."""
    from torchvision.transforms import functional as TF

    def apply(x: torch.Tensor) -> torch.Tensor:
        return TF.rotate(x, angle_degrees)

    return apply


def permuted_tasks(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_tasks: int = 5,
    seed: int = 0,
    identity_first: bool = True,
    calibration_fraction: float = 0.1,
    min_calibration: int = 500,
    batch_size: int = 64,
    num_workers: int = 0,
    test_batch_size: Optional[int] = None,
) -> List[TaskData]:
    """Fixed random input permutation per task (Permuted MNIST-style).

    Task 0 keeps the identity permutation by default, as in the original
    Permuted MNIST protocol.
    """
    sample, _ = train_dataset[0]
    shape = tuple(sample.shape)
    dimension = int(sample.numel())
    generator = torch.Generator().manual_seed(seed)
    transforms = []
    for task_id in range(num_tasks):
        if task_id == 0 and identity_first:
            permutation = torch.arange(dimension)
        else:
            permutation = torch.randperm(dimension, generator=generator)
        transforms.append(permutation_transform(permutation, shape))
    return domain_incremental_tasks(
        train_dataset,
        test_dataset,
        transforms,
        calibration_fraction=calibration_fraction,
        min_calibration=min_calibration,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        test_batch_size=test_batch_size,
    )


def contiguous_class_groups(num_classes: int, classes_per_task: int) -> List[List[int]]:
    """Class-incremental split into contiguous groups, e.g. 5 x 2 for MNIST."""
    if num_classes % classes_per_task != 0:
        raise ValueError("num_classes must be divisible by classes_per_task")
    return [
        list(range(start, start + classes_per_task))
        for start in range(0, num_classes, classes_per_task)
    ]


def num_classes(dataset: Dataset) -> int:
    """Number of distinct labels in a labeled dataset."""
    return max(int(value) for value in _targets(dataset)) + 1


def mnist_datasets(root: str):
    """MNIST (train, test) with the standard normalisation (downloads to ``root``)."""
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
    return train, test


def cifar10_datasets(root: str):
    """CIFAR-10 (train, test) with the standard normalisation (downloads)."""
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root, train=False, download=True, transform=transform)
    return train, test


def cifar100_datasets(root: str):
    """CIFAR-100 (train, test) with the standard normalisation (downloads)."""
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    train = datasets.CIFAR100(root, train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root, train=False, download=True, transform=transform)
    return train, test


def split_mnist(root: str, classes_per_task: int = 2, **kwargs) -> List[TaskData]:
    """Split MNIST into class-incremental tasks (downloads to ``root``)."""
    train, test = mnist_datasets(root)
    return class_incremental_tasks(
        train, test, contiguous_class_groups(10, classes_per_task), **kwargs
    )


def rotated_mnist(
    root: str, num_tasks: int = 5, angles: Optional[Sequence[float]] = None, **kwargs
) -> List[TaskData]:
    """Rotated MNIST: domain-incremental tasks, evenly spaced rotations."""
    train, test = mnist_datasets(root)
    angles = list(angles) if angles is not None else [
        index * 360.0 / num_tasks for index in range(num_tasks)
    ]
    return domain_incremental_tasks(
        train, test, [rotation_transform(angle) for angle in angles], **kwargs
    )


def permuted_mnist(
    root: str, num_tasks: int = 5, seed: int = 0, **kwargs
) -> List[TaskData]:
    """Permuted MNIST: one fixed random pixel permutation per task."""
    train, test = mnist_datasets(root)
    return permuted_tasks(train, test, num_tasks=num_tasks, seed=seed, **kwargs)


def split_cifar100(root: str, classes_per_task: int = 10, **kwargs) -> List[TaskData]:
    """Split CIFAR-100 into class-incremental tasks (downloads to ``root``)."""
    train, test = cifar100_datasets(root)
    return class_incremental_tasks(
        train, test, contiguous_class_groups(100, classes_per_task), **kwargs
    )


def split_cifar10(root: str, classes_per_task: int = 2, **kwargs) -> List[TaskData]:
    """Split CIFAR-10 into class-incremental tasks (downloads to ``root``)."""
    train, test = cifar10_datasets(root)
    return class_incremental_tasks(
        train, test, contiguous_class_groups(10, classes_per_task), **kwargs
    )


class LabelledTensorDataset(TensorDataset):
    """TensorDataset with a ``targets`` attribute, for the splitters."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__(features, labels)
        self.targets = labels.tolist()


def synthetic_pair(
    num_classes: int = 6,
    dimension: int = 8,
    samples_per_class: int = 150,
    separation: float = 3.0,
    seed: int = 0,
):
    """Gaussian class clusters as (train, test) datasets, for tests/smoke runs."""
    generator = torch.Generator().manual_seed(seed)
    centres = separation * torch.randn(num_classes, dimension, generator=generator)
    features = []
    labels = []
    for label in range(num_classes):
        features.append(
            centres[label] + torch.randn(samples_per_class, dimension, generator=generator)
        )
        labels.append(torch.full((samples_per_class,), label, dtype=torch.long))
    x = torch.cat(features)
    y = torch.cat(labels)
    dataset = LabelledTensorDataset(x, y)
    return dataset, dataset


def synthetic_tasks(
    num_classes: int = 6,
    classes_per_task: int = 2,
    dimension: int = 8,
    samples_per_class: int = 150,
    separation: float = 3.0,
    seed: int = 0,
    **split_kwargs,
) -> List[TaskData]:
    """Gaussian class clusters, for local tests and smoke runs."""
    dataset, _ = synthetic_pair(
        num_classes=num_classes,
        dimension=dimension,
        samples_per_class=samples_per_class,
        separation=separation,
        seed=seed,
    )
    groups = [
        list(range(start, start + classes_per_task))
        for start in range(0, num_classes, classes_per_task)
    ]
    return class_incremental_tasks(dataset, dataset, groups, seed=seed, **split_kwargs)
