"""Method plugin interface (gmumcl's plugin shape, extended for task ends)."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class MethodPluginABC(abc.ABC):
    """A method hooks into training through these calls.

    ``set_model`` and ``setup_task`` happen before a task is trained,
    ``forward`` composes the loss of every batch, ``on_after_backward`` runs
    between ``loss.backward()`` and the optimiser step (gradient projection,
    path integrals), and ``on_task_end`` finalises the task (statistics,
    region fitting, projector updates). The remaining hooks let a method
    override its predictor and report diagnostics and figures.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.model: Optional[nn.Module] = None

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def setup_task(self, task_id: int, task: Optional[Any] = None) -> None:
        """Called before training task ``task_id`` (0-based), with its ``TaskData``."""

    def on_before_forward(self) -> None:
        """Called once per batch, after ``model.train()`` (e.g. keep BN in eval)."""

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-class scores used for prediction (the logits by default).

        Methods whose predictor is not the linear head override this; the
        trainer and every accuracy metric read predictions through this
        callable, so the evaluation protocol is method-independent.
        """
        if self.model is None:
            raise RuntimeError("scores called before set_model")
        return self.model(x)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the composed loss and logits for one batch."""
        return loss, logits

    def on_after_backward(self) -> None:
        """Called after ``loss.backward()`` and before the optimiser step."""

    def on_task_end(self, task_id: int, task: Any) -> None:
        """Called once the task's training loop is finished."""

    def diagnostics(self) -> Dict[str, Any]:
        """Method-specific run diagnostics, merged into the run's metrics."""
        return {}

    def figures(self, output_dir: str, config: Dict[str, Any], tasks: List) -> List[str]:
        """Write this method's run figures into ``output_dir``; returns their paths."""
        return []

    def figure_payload(self) -> Dict[str, Any]:
        """Raw, model-free data from which selected runs redraw their figures."""
        return {}
