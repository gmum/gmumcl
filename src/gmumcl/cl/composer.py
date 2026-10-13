"""Loss composer: model + criterion + plugins, one batch at a time."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .plugin import MethodPluginABC


class Composer:
    """Runs one training batch: model, task loss, plugin losses, backward."""

    def __init__(self, model: nn.Module, criterion: nn.Module, plugins: List[MethodPluginABC]) -> None:
        self.model = model
        self.criterion = criterion
        self.plugins = plugins

    def run_batch(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Forward, compose losses, backward, run plugin post-backward hooks, step."""
        self.model.train()
        for plugin in self.plugins:
            plugin.on_before_forward()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        for plugin in self.plugins:
            loss, logits = plugin.forward(x, y, loss, logits)
        optimizer.zero_grad()
        loss.backward()
        for plugin in self.plugins:
            plugin.on_after_backward()
        optimizer.step()
        return float(loss.detach())
