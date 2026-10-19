"""Sequential trainer: tasks, plugins, accuracy matrix."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .composer import Composer
from .metrics import AccuracyMatrix, new_class_margin, restricted_accuracy, task_accuracy
from .plugin import MethodPluginABC
from .scenarios import TaskData


@dataclass
class TrainerConfig:
    epochs: int = 1
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    log_every: int = 0
    optimizer: str = "adam"
    momentum: float = 0.0
    iterations: Optional[int] = None
    verbose: bool = False
    task_oracle: bool = False
    """Mask the scores to the evaluated task's classes (task-incremental)."""


class Trainer:
    """Trains a task sequence and returns the accuracy matrix R."""

    def __init__(
        self,
        model: nn.Module,
        plugins: List[MethodPluginABC],
        config: Optional[TrainerConfig] = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = self.config.device
        self.model = model.to(self.device)
        self.plugins = plugins
        self.criterion = nn.CrossEntropyLoss()
        self.composer = Composer(self.model, self.criterion, self.plugins)
        for plugin in self.plugins:
            plugin.set_model(self.model)
        self.scores = self.plugins[0].scores if self.plugins else self.model

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(parameters, lr=self.config.lr, momentum=self.config.momentum)
        return torch.optim.Adam(parameters, lr=self.config.lr)

    def _train_task(
        self,
        task: TaskData,
        optimizer: torch.optim.Optimizer,
        on_step: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Epoch-based, or exactly ``config.iterations`` steps per task (cycling).

        ``on_step`` (diagnostics) is called with the 1-based global step count.
        """
        step = 0
        if self.config.iterations is None:
            for epoch in range(self.config.epochs):
                start = time.time()
                for x, y in task.fit_loader:
                    self.composer.run_batch(x.to(self.device), y.to(self.device), optimizer)
                    step += 1
                    if on_step is not None:
                        on_step(step)
                if self.config.verbose:
                    print(
                        "task %d epoch %d/%d %.1fs"
                        % (task.task_id, epoch + 1, self.config.epochs, time.time() - start),
                        flush=True,
                    )
            return
        while step < self.config.iterations:
            for x, y in task.fit_loader:
                self.composer.run_batch(x.to(self.device), y.to(self.device), optimizer)
                step += 1
                if on_step is not None:
                    on_step(step)
                if step >= self.config.iterations:
                    break

    def run(self, tasks: List[TaskData]) -> AccuracyMatrix:
        matrix = AccuracyMatrix()
        seen_classes: List[int] = []
        cumulative: List[List[int]] = []
        for position, task in enumerate(tasks):
            seen_classes.extend(task.classes)
            previous = cumulative[-1] if cumulative else []
            cumulative.append(sorted(set(previous) | set(task.classes)))
            self.model.expand_head(max(seen_classes) + 1)
            if self.config.verbose:
                print("task %d/%d start" % (position + 1, len(tasks)), flush=True)
            for plugin in self.plugins:
                plugin.setup_task(task.task_id, task)
            optimizer = self._make_optimizer()
            self._train_task(task, optimizer)

            for plugin in self.plugins:
                plugin.on_task_end(task.task_id, task)

            self.model.eval()
            matrix.append(
                [
                    task_accuracy(
                        self.scores,
                        tasks[s].test_loader,
                        self.device,
                        classes=tasks[s].classes,
                        mask_logits=self.config.task_oracle,
                    )
                    for s in range(position + 1)
                ]
            )
            matrix.append_old_class(
                [
                    restricted_accuracy(self.scores, tasks[s].test_loader, self.device, cumulative[s])
                    for s in range(position + 1)
                ]
            )
            matrix.append_task_aware(
                [
                    restricted_accuracy(self.scores, tasks[s].test_loader, self.device, tasks[s].classes)
                    for s in range(position + 1)
                ]
            )
            matrix.append_new_class_margins(
                [
                    new_class_margin(
                        self.scores,
                        tasks[s].test_loader,
                        self.device,
                        [c for c in cumulative[position] if c not in cumulative[s]],
                    )
                    if s < position
                    else None
                    for s in range(position + 1)
                ]
            )
            if any(task.validation_loader is not None for task in tasks):
                matrix.append_validation(
                    [
                        task_accuracy(
                            self.scores,
                            tasks[s].validation_loader,
                            self.device,
                            classes=tasks[s].classes,
                            mask_logits=self.config.task_oracle,
                        )
                        if tasks[s].validation_loader is not None
                        else None
                        for s in range(position + 1)
                    ]
                )
        return matrix
