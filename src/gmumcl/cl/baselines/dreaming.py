"""Dreaming: generative replay from the RBF head's class kernels.

Port of `OneAndZero24/local-cl` `src/method/dreaming.py` (MIT), adapted to
the plugin API:

- at the start of every task after the first, the previous model's RBF head
  (one Gaussian kernel per old class: centre and per-feature shape) defines
  the target statistics; a batch of synthetic inputs per class is optimised
  until the frozen teacher's penultimate activations match them (mean and
  variance match, relaxed Mahalanobis pull, L2 and total-variation priors,
  pairwise-diversity bonus),
- during the task, the current model distils the teacher's penultimate
  activations on the dreamed inputs (mean squared error).

The method needs a model with ``head_type="rbf"`` (see `gmumcl.cl.rbf`).
Documented deviation: distillation is the mean squared feature error, not the
unnormalised L2 norm of the original.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..plugin import MethodPluginABC
from ..rbf import RBFHead


def _total_variation(images: torch.Tensor) -> torch.Tensor:
    """Smoothness prior for image inputs; zero for vector inputs."""
    if images.dim() != 4:
        return torch.zeros((), dtype=images.dtype, device=images.device)
    horizontal = (images[:, :, :-1, :] - images[:, :, 1:, :]).abs().mean()
    vertical = (images[:, :, :, :-1] - images[:, :, :, 1:]).abs().mean()
    return horizontal + vertical


class Dreaming(MethodPluginABC):
    """Feature-space replay driven by the RBF head (see module docstring)."""

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.alpha = float(self.config.get("alpha", 1.0))
        self.samples_per_class = int(self.config.get("samples_per_class", 32))
        self.steps = int(self.config.get("steps", 100))
        self.dreaming_lr = float(self.config.get("dreaming_lr", 0.1))
        self.lambda_l2 = float(self.config.get("lambda_l2", 0.01))
        self.lambda_tv = float(self.config.get("lambda_tv", 0.001))
        self.diversity_weight = float(self.config.get("diversity_weight", 0.05))
        self.in_shape: Tuple[int, ...] = tuple(self.config.get("in_shape", (1, 28, 28)))
        self.seen_classes = 0
        self.dreamed: Dict[int, torch.Tensor] = {}
        self.old_model: Optional[nn.Module] = None
        self.task_id = 0

    def set_model(self, model: nn.Module) -> None:
        super().set_model(model)
        if not isinstance(getattr(model, "head", None), RBFHead):
            raise TypeError(
                "Dreaming requires an RBF head; set model.head_type='rbf'"
            )

    def setup_task(self, task_id: int, task: Optional[Any] = None) -> None:
        self.task_id = task_id
        self.dreamed = {}
        self.old_model = None
        if task_id == 0 or self.seen_classes == 0:
            return
        assert self.model is not None
        head = getattr(self.model, "head")
        centers = head.kernels_centers[: self.seen_classes].detach().clone()
        shapes = head.log_shapes[: self.seen_classes].exp().detach().clone()
        self.old_model = copy.deepcopy(self.model).eval()
        for parameter in self.old_model.parameters():
            parameter.requires_grad = False
        self._generate(centers, shapes)

    def _penultimate(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        captured: Dict[str, torch.Tensor] = {}
        head = getattr(model, "head")
        handle = head.register_forward_pre_hook(
            lambda module, inputs: captured.__setitem__("value", inputs[0])
        )
        try:
            model(x)
        finally:
            handle.remove()
        return captured["value"]

    def _generate(self, centers: torch.Tensor, shapes: torch.Tensor) -> None:
        """Optimise one batch of inputs per class to match the old kernels."""
        assert self.model is not None and self.old_model is not None
        device = next(self.model.parameters()).device
        for class_index in range(centers.shape[0]):
            images = torch.randn(
                (self.samples_per_class, *self.in_shape), device=device, requires_grad=True
            )
            optimizer = torch.optim.Adam([images], lr=self.dreaming_lr)
            center = centers[class_index].to(device)
            shape = shapes[class_index].to(device)
            for _ in range(self.steps):
                optimizer.zero_grad()
                features = self._penultimate(self.old_model, images)
                mean_loss = F.mse_loss(features.mean(dim=0), center)
                variance_loss = F.mse_loss(features.var(dim=0, unbiased=False), shape)
                difference = features - center
                mahalanobis = torch.mean((difference / (shape + 1e-6)) ** 2)
                l2 = images.norm(p=2) / self.samples_per_class
                diversity = -torch.mean(
                    torch.norm(
                        features.unsqueeze(1) - features.unsqueeze(0), p=2, dim=-1
                    )
                )
                total = (
                    mean_loss
                    + variance_loss
                    + mahalanobis
                    + self.lambda_l2 * l2
                    + self.lambda_tv * _total_variation(images)
                    + self.diversity_weight * diversity
                )
                total.backward()
                optimizer.step()
            self.dreamed[class_index] = images.detach()

    def forward(self, x, y, loss, logits):
        if not self.dreamed or self.old_model is None or self.model is None:
            return loss, logits
        total = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for images in self.dreamed.values():
            images = images.to(loss.device)
            old_features = self._penultimate(self.old_model, images).detach()
            current_features = self._penultimate(self.model, images)
            total = total + F.mse_loss(current_features, old_features)
        total = total / max(len(self.dreamed), 1)
        return loss + self.alpha * total, logits

    def on_task_end(self, task_id: int, task) -> None:
        self.seen_classes += len(task.classes)
