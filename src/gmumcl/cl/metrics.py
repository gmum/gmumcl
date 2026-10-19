"""Continual-learning metrics.

Every accuracy helper takes a ``scores`` callable mapping a batch to per-class
scores. The plain model is such a callable, and a method can route evaluation
through its own predictor via :meth:`MethodPluginABC.scores`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@torch.no_grad()
def task_accuracy(
    scores,
    loader,
    device: str,
    classes: Optional[List[int]] = None,
    mask_logits: bool = False,
) -> float:
    """Top-1 accuracy on a loader, optionally restricted to ``classes``.

    With ``mask_logits`` the scores are restricted to the ``classes`` columns
    (the task oracle of task-incremental evaluation) and the labels are
    re-indexed to those columns; without it the full head is used and only
    samples of those classes are scored (class-incremental).
    """
    correct = 0
    total = 0
    allowed = None if classes is None else torch.tensor(sorted(classes))
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = scores(x)
        if allowed is not None:
            mask = torch.isin(y.cpu(), allowed)
            if not bool(mask.any()):
                continue
            x, y, logits = x[mask.to(x.device)], y[mask.to(x.device)], logits[mask.to(x.device)]
            if mask_logits:
                positions = torch.searchsorted(allowed, y.cpu())
                logits = logits[:, allowed.to(logits.device)]
                y = positions.to(y.device)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def restricted_accuracy(scores, loader, device: str, classes) -> float:
    """Top-1 accuracy on a loader with the argmax restricted to ``classes``.

    Used for the task-aware accuracy (``classes`` = the task's own classes) and
    the old-class accuracy (``classes`` = every class seen up to that task).
    """
    allowed = torch.tensor(sorted({int(value) for value in classes}), dtype=torch.long)
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = scores(x)
        index = allowed.to(logits.device)
        predictions = index[logits[:, index].argmax(dim=1)]
        correct += int((predictions == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def new_class_margin(scores, loader, device: str, new_classes) -> float:
    """Mean of (max score over ``new_classes`` - correct-class score)."""
    classes = sorted({int(value) for value in new_classes})
    if not classes:
        return 0.0
    index = torch.tensor(classes, dtype=torch.long)
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = scores(x)
        new_max = logits[:, index.to(logits.device)].max(dim=1).values
        correct = logits.gather(1, y[:, None]).squeeze(1)
        total += float((new_max - correct).sum())
        count += int(y.numel())
    return total / max(count, 1)


@dataclass
class AccuracyMatrix:
    """R[t][s] = accuracy on task s after training task t.

    ``old_class_rows`` restricts the argmax to the classes seen up to task s
    (argmax over the old head rather than the full head),
    ``task_aware_rows`` to the task's own classes, ``new_class_margin_rows``
    records the mean new-class score margin on old tasks, and
    ``validation_rows`` holds validation accuracies where a validation loader
    exists.
    """

    rows: List[List[float]] = field(default_factory=list)
    old_class_rows: List[List[float]] = field(default_factory=list)
    task_aware_rows: List[List[float]] = field(default_factory=list)
    new_class_margin_rows: List[List[Optional[float]]] = field(default_factory=list)
    validation_rows: List[List[Optional[float]]] = field(default_factory=list)

    def append(self, row: List[float]) -> None:
        self.rows.append(list(row))

    def append_old_class(self, row: List[float]) -> None:
        self.old_class_rows.append(list(row))

    def append_task_aware(self, row: List[float]) -> None:
        self.task_aware_rows.append(list(row))

    def append_new_class_margins(self, row: List[Optional[float]]) -> None:
        self.new_class_margin_rows.append(list(row))

    def append_validation(self, row: List[Optional[float]]) -> None:
        self.validation_rows.append(list(row))

    @property
    def num_tasks(self) -> int:
        return len(self.rows)

    def old_accuracy(self, t: int) -> float:
        row = self.rows[t]
        return sum(row[:t]) / max(t, 1)

    def new_accuracy(self, t: int) -> float:
        return self.rows[t][t]

    def forgetting(self) -> float:
        if self.num_tasks < 2:
            return 0.0
        drops = []
        for s in range(self.num_tasks - 1):
            best = max(self.rows[t][s] for t in range(s, self.num_tasks))
            drops.append(best - self.rows[-1][s])
        return sum(drops) / max(len(drops), 1)

    def backward_transfer(self) -> float:
        if self.num_tasks < 2:
            return 0.0
        values = [self.rows[self.num_tasks - 1][s] - self.rows[s][s] for s in range(self.num_tasks - 1)]
        return sum(values) / max(len(values), 1)

    def forward_transfer(self) -> float:
        """Change of new-task accuracy from one task to the next (proxy for FWT)."""
        if self.num_tasks < 2:
            return 0.0
        values = [self.rows[s][s] - self.rows[s - 1][s - 1] for s in range(1, self.num_tasks)]
        return sum(values) / max(len(values), 1)

    def average_incremental_accuracy(self) -> float:
        means = [sum(self.rows[t][: t + 1]) / (t + 1) for t in range(self.num_tasks)]
        return sum(means) / max(len(means), 1)

    def final_old_new(self) -> tuple:
        last = self.num_tasks - 1
        if last < 1:
            return 0.0, self.rows[last][last]
        return self.old_accuracy(last), self.new_accuracy(last)

    def final_old_class_accuracy(self) -> float:
        """Mean old-class accuracy (argmax over classes seen up to each task)."""
        last = self.num_tasks - 1
        if last < 1:
            return 0.0
        row = self.old_class_rows[last]
        return sum(row[:last]) / last

    def final_average_accuracy(self) -> float:
        """Mean test accuracy over all tasks after the last task."""
        row = self.rows[-1]
        return sum(row) / max(len(row), 1)

    def validation_final_average_accuracy(self) -> Optional[float]:
        """Mean validation accuracy over all tasks after the last task."""
        if not self.validation_rows:
            return None
        values = [value for value in self.validation_rows[-1] if value is not None]
        if not values:
            return None
        return sum(values) / len(values)
