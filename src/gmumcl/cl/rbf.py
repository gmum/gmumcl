"""Gaussian RBF classification head with a growing class axis.

Lineage: ``SingleRBFHeadLayer`` of `OneAndZero24/local-cl` (MIT), itself
based on Alessio Russo's PyTorchRBFLayer (MIT). Each class owns one kernel:
a centre and a per-feature log-shape in the penultimate feature space, so the
head parameters double as the per-class statistics Dreaming replays.

The forward pass returns the (unnormalised) Gaussian kernel activations,
one column per class; ``CrossEntropyLoss`` turns them into logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RBFHead(nn.Module):
    """One Gaussian kernel per class, grown class by class."""

    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.kernels_centers = nn.Parameter(torch.empty(n_classes, in_features))
        self.log_shapes = nn.Parameter(torch.empty(n_classes, in_features))
        self.reset()

    def reset(self) -> None:
        """Uniform centres and mildly spread log-shapes (official init)."""
        nn.init.uniform_(self.kernels_centers, a=0.0, b=1.0)
        nn.init.normal_(self.log_shapes, mean=0.5, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centers = self.kernels_centers.unsqueeze(0)
        shapes = self.log_shapes.exp().unsqueeze(0)
        difference = (x.unsqueeze(1) - centers) / shapes
        return (-(difference**2).sum(dim=-1)).exp()

    @torch.no_grad()
    def expand_head(self, n_classes: int) -> None:
        """Append kernels for new classes, keeping the learned old rows."""
        if n_classes <= self.n_classes:
            return
        device, dtype = self.kernels_centers.device, self.kernels_centers.dtype
        centers = torch.empty(n_classes, self.in_features, device=device, dtype=dtype)
        shapes = torch.empty(n_classes, self.in_features, device=device, dtype=dtype)
        nn.init.uniform_(centers, a=0.0, b=1.0)
        nn.init.normal_(shapes, mean=0.5, std=0.1)
        centers[: self.n_classes] = self.kernels_centers
        shapes[: self.n_classes] = self.log_shapes
        self.kernels_centers = nn.Parameter(centers)
        self.log_shapes = nn.Parameter(shapes)
        self.n_classes = n_classes
