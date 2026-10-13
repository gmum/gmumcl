"""Method plugin interface (gmumcl's plugin shape, extended for task ends)."""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class MethodPluginABC(abc.ABC):
    """A method hooks into training through these five calls.

    ``set_model`` and ``setup_task`` happen before a task is trained,
    ``forward`` composes the loss of every batch, ``on_after_backward`` runs
    between ``loss.backward()`` and the optimiser step (gradient projection,
    path integrals), and ``on_task_end`` finalises the task (statistics,
    region fitting, projector updates).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.model: Optional[nn.Module] = None

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def setup_task(self, task_id: int) -> None:
        """Called before training task ``task_id`` (0-based)."""

    def on_before_forward(self) -> None:
        """Called once per batch, after ``model.train()`` (e.g. keep BN in eval)."""

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the composed loss and logits for one batch."""
        return loss, logits

    def on_after_backward(self) -> None:
        """Called after ``loss.backward()`` and before the optimiser step."""

    def on_task_end(self, task_id: int, task: Any) -> None:
        """Called once the task's training loop is finished."""
