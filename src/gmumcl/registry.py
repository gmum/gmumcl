"""Config-driven builders: `name` keys map to builders (dict lookups, no ifs)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from .cl.baselines import build as build_baseline
from .cl.models import MLP, ResNet18, SmallCNN, ViT
from .cl.scenarios import (
    cifar10_datasets,
    cifar100_datasets,
    class_incremental_tasks,
    contiguous_class_groups,
    domain_incremental_tasks,
    instance_incremental_tasks,
    mnist_datasets,
    num_classes,
    permuted_tasks,
    rotation_transform,
    shift_transform,
    synthetic_pair,
    task_incremental_tasks,
)

SCENARIOS = (
    "class_incremental",
    "task_incremental",
    "domain_incremental",
    "instance_incremental",
    "permuted",
)
"""Scenario keys accepted by ``data.scenario``."""


def scenario_of(data: Dict[str, Any]) -> str:
    """Scenario of a data section; the dataset names imply their scenario."""
    name = data.get("name", "toy")
    if name == "permuted_mnist":
        return "permuted"
    if name == "rotated_mnist":
        return "domain_incremental"
    return str(data.get("scenario", "class_incremental"))


def _split_kwargs(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Common fitting/calibration split settings of every task source."""
    return {
        "calibration_fraction": float(data.get("calibration_fraction", 0.1)),
        "min_calibration": int(data.get("min_calibration", 50)),
        "batch_size": int(data.get("batch_size", 32)),
        "seed": int(config.get("seed", 0)),
    }


def _toy_pair(data: Dict[str, Any], config: Dict[str, Any]):
    return synthetic_pair(
        num_classes=int(data.get("num_classes", 6)),
        dimension=int(data.get("dimension", 8)),
        samples_per_class=int(data.get("samples_per_class", 150)),
        separation=float(data.get("separation", 3.0)),
        seed=int(config.get("seed", 0)),
    )


def _mnist_pair(data: Dict[str, Any], config: Dict[str, Any]):
    return mnist_datasets(data.get("root", "data"))


def _cifar10_pair(data: Dict[str, Any], config: Dict[str, Any]):
    return cifar10_datasets(data.get("root", "data"))


def _cifar100_pair(data: Dict[str, Any], config: Dict[str, Any]):
    return cifar100_datasets(data.get("root", "data"))


DATASET_PAIRS = {
    "toy": _toy_pair,
    "mnist": _mnist_pair,
    "cifar10": _cifar10_pair,
    "cifar100": _cifar100_pair,
    "permuted_mnist": _mnist_pair,
    "rotated_mnist": _mnist_pair,
}
"""Dataset name -> builder returning a (train, test) dataset pair."""


def _domain_tasks(
    train, test, data: Dict[str, Any], config: Dict[str, Any], **kwargs
) -> List:
    """Generic domain-incremental split: rotations for images, shifts for vectors."""
    sample = train[0][0]
    num_tasks = int(data.get("num_tasks", 5))
    if sample.dim() == 3:
        angles = data.get("angles")
        if angles is None:
            angles = [index * 360.0 / num_tasks for index in range(num_tasks)]
        transforms = [rotation_transform(float(angle)) for angle in angles]
    else:
        generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
        scale = float(data.get("shift_scale", 1.0))
        shifts = scale * torch.randn(num_tasks, sample.numel(), generator=generator)
        transforms = [shift_transform(shift) for shift in shifts]
    return domain_incremental_tasks(train, test, transforms, **kwargs)


def build_tasks(config: Dict[str, Any]) -> List:
    """Task list for ``data.name`` and ``data.scenario`` (KeyError lists options)."""
    data = config.get("data", {})
    name = data.get("name", "toy")
    builder = DATASET_PAIRS.get(name)
    if builder is None:
        raise KeyError("unknown dataset %r; available: %s" % (name, sorted(DATASET_PAIRS)))
    scenario = scenario_of(data)
    if scenario not in SCENARIOS:
        raise KeyError("unknown scenario %r; available: %s" % (scenario, list(SCENARIOS)))
    train, test = builder(data, config)
    kwargs = _split_kwargs(data, config)

    if scenario in ("class_incremental", "task_incremental"):
        groups = contiguous_class_groups(
            num_classes(train), int(data.get("classes_per_task", 2))
        )
        source = class_incremental_tasks if scenario == "class_incremental" else task_incremental_tasks
        return source(train, test, groups, **kwargs)
    if scenario == "instance_incremental":
        return instance_incremental_tasks(
            train, test, num_tasks=int(data.get("num_tasks", 5)), **kwargs
        )
    if scenario == "permuted":
        kwargs.pop("seed", None)
        return permuted_tasks(
            train,
            test,
            num_tasks=int(data.get("num_tasks", 5)),
            seed=int(config.get("seed", 0)),
            identity_first=bool(data.get("identity_first", True)),
            **kwargs,
        )
    return _domain_tasks(train, test, data, config, **kwargs)


MODEL_BUILDERS = {
    "mlp": lambda model, config, n_classes: MLP(
        input_dim=int(model.get("input_dim", config.get("data", {}).get("dimension", 8))),
        n_classes=n_classes,
        hidden=int(model.get("hidden", 64)),
        layers=int(model.get("layers", 2)),
        head_type=str(model.get("head_type", "linear")),
    ),
    "small_cnn": lambda model, config, n_classes: SmallCNN(
        n_classes=n_classes,
        in_channels=int(model.get("in_channels", 3)),
        head_type=str(model.get("head_type", "linear")),
    ),
    "resnet18": lambda model, config, n_classes: ResNet18(
        n_classes=n_classes,
        in_channels=int(model.get("in_channels", 3)),
        pretrained=bool(model.get("pretrained", False)),
    ),
    "vit": lambda model, config, n_classes: ViT(
        n_classes=n_classes,
        pretrained=bool(model.get("pretrained", False)),
        freeze_backbone=bool(model.get("freeze_backbone", False)),
    ),
}


def build_model(config: Dict[str, Any], n_classes: int) -> torch.nn.Module:
    """Model instance for ``model.name`` (KeyError lists the known architectures)."""
    model = config.get("model", {})
    name = model.get("name", "mlp")
    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        raise KeyError("unknown model %r; available: %s" % (name, sorted(MODEL_BUILDERS)))
    return builder(model, config, n_classes)


def build_plugins(config: Dict[str, Any]) -> list:
    """Plugin list for ``method.name``; unknown names list the baselines."""
    method = config.get("method", {})
    name = method.get("name", "naive")
    settings = {key: value for key, value in method.items() if key != "name"}
    return [build_baseline(name, settings)]
