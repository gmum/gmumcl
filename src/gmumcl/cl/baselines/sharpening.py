"""Sharpening: push the most active units further and damp the rest.

Port of `OneAndZero24/local-cl` `src/method/sharpening.py` (MIT), after the
original sharpening idea (AAAI Spring Symposium 1993). The plugin accumulates
the task's mean affine activations, picks the top-K coordinates and builds a
sharpened target: top coordinates move towards ``gamma * (1 - a)``, the rest
shrink by ``gamma``.

Documented deviation: the original target was detached, so the added term had
no gradient (training reduced to rescaling the task loss by ``alpha``). This
port keeps the accumulated buffer as the target but penalises the *current*
batch's mean activations against it, so the term is differentiable and the
method does what its name says.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..plugin import MethodPluginABC


class Sharpening(MethodPluginABC):
    """Top-K activation sharpening against a running task-average buffer."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.alpha = float(self.config.get("alpha", 0.5))
        self.gamma = float(self.config.get("gamma", 0.1))
        self.top_k = int(self.config.get("K", 64))
        self.activation_buffer = None
        self._captured: List[torch.Tensor] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._capturing = False

    def set_model(self, model: nn.Module) -> None:
        super().set_model(model)
        for _, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._handles.append(module.register_forward_hook(self._capture))

    def _capture(self, module, inputs, output) -> None:
        if self._capturing:
            self._captured.append(output)

    def setup_task(self, task_id: int, task: Optional[Any] = None) -> None:
        self.activation_buffer = None

    def on_before_forward(self) -> None:
        self._captured = []
        self._capturing = True

    def _current_activations(self):
        if not self._captured:
            return None
        return torch.cat(
            [activation.mean(dim=0).reshape(-1) for activation in self._captured]
        )

    def forward(self, x, y, loss, logits):
        self._capturing = False
        activation = self._current_activations()
        if activation is None:
            return loss, logits
        if self.activation_buffer is None:
            self.activation_buffer = torch.zeros_like(activation.detach())
        self.activation_buffer = self.activation_buffer + activation.detach()

        count = min(self.top_k, self.activation_buffer.numel())
        top = torch.topk(self.activation_buffer.abs(), count).indices
        target = self.activation_buffer.clone()
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[top] = True
        target[mask] = self.gamma * (1.0 - target[mask])
        target[~mask] = (1.0 - self.gamma) * target[~mask]

        sharpening = F.mse_loss(activation, target.detach())
        return self.alpha * loss + (1.0 - self.alpha) * sharpening, logits
