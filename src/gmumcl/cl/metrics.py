"""Continual-learning metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def task_accuracy(
    model: nn.Module,
    loader,
    device: str,
    classes: Optional[List[int]] = None,
    mask_logits: bool = False,
) -> float:
    """Top-1 accuracy on a loader, optionally restricted to ``classes``.

    With ``mask_logits`` the logits are restricted to the ``classes`` columns
    (the task oracle of task-incremental evaluation); without it the full head
    is used and only samples of those classes are scored (class-incremental).
    """
    model.eval()
    correct = 0
    total = 0
    allowed = None if classes is None else torch.tensor(sorted(classes))
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
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


@dataclass
class AccuracyMatrix:
    """R[t][s] = accuracy on task s after training task t."""

    rows: List[List[float]] = field(default_factory=list)

    def append(self, row: List[float]) -> None:
        self.rows.append(list(row))

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
        means = [
            sum(self.rows[t][: t + 1]) / (t + 1) for t in range(self.num_tasks)
        ]
        return sum(means) / max(len(means), 1)

    def final_old_new(self) -> tuple:
        last = self.num_tasks - 1
        if last < 1:
            return 0.0, self.rows[last][last]
        return self.old_accuracy(last), self.new_accuracy(last)
