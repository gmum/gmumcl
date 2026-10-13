"""Models for the controlled experiments: MLP, SmallCNN, ResNet18 and ViT.

Every model exposes ``expand_head`` (grow the classification head per task
while keeping the learned rows) and ``n_classes``. ``head_type="rbf"`` swaps
the linear head of the MLP and SmallCNN for :class:`~gmumcl.cl.rbf.RBFHead`,
which the Dreaming method replays.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .rbf import RBFHead

PROTECTED_TYPES = (nn.Linear, nn.Conv2d)


def _grow_linear_head(head: nn.Linear, n_classes: int) -> nn.Linear:
    """A larger linear head with the old rows copied into the leading rows."""
    new_head = nn.Linear(head.in_features, n_classes).to(head.weight)
    with torch.no_grad():
        new_head.weight[: head.out_features] = head.weight
        new_head.bias[: head.out_features] = head.bias
    return new_head


class MLP(nn.Module):
    """Fully connected network with a growing classification head."""

    def __init__(
        self,
        input_dim: int = 784,
        n_classes: int = 10,
        hidden: int = 400,
        layers: int = 2,
        head_type: str = "linear",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        modules: List[nn.Module] = []
        width = input_dim
        for _ in range(layers):
            modules.append(nn.Linear(width, hidden))
            modules.append(nn.ReLU())
            width = hidden
        self.features = nn.Sequential(*modules)
        self.head = RBFHead(width, n_classes) if head_type == "rbf" else nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.flatten(1)))

    def expand_head(self, n_classes: int) -> None:
        """Grow the head, keeping the learned rows of earlier classes."""
        if n_classes <= self.n_classes:
            return
        if isinstance(self.head, RBFHead):
            self.head.expand_head(n_classes)
        else:
            self.head = _grow_linear_head(self.head, n_classes)
        self.n_classes = n_classes


class SmallCNN(nn.Module):
    """Two convolutional blocks plus a head; no BatchNorm (certified)."""

    def __init__(self, n_classes: int = 10, in_channels: int = 3, head_type: str = "linear") -> None:
        super().__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = RBFHead(64, n_classes) if head_type == "rbf" else nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))

    def expand_head(self, n_classes: int) -> None:
        if n_classes <= self.n_classes:
            return
        if isinstance(self.head, RBFHead):
            self.head.expand_head(n_classes)
        else:
            self.head = _grow_linear_head(self.head, n_classes)
        self.n_classes = n_classes


class ResNet18(nn.Module):
    """torchvision ResNet-18 with a growable head."""

    def __init__(self, n_classes: int = 100, in_channels: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(512, n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def expand_head(self, n_classes: int) -> None:
        if n_classes <= self.n_classes:
            return
        self.head = _grow_linear_head(self.head, n_classes)
        self.n_classes = n_classes


class ViT(nn.Module):
    """torchvision ViT-B/16 with a growable head; the backbone can be frozen."""

    def __init__(self, n_classes: int = 200, pretrained: bool = False, freeze_backbone: bool = False) -> None:
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        network = vit_b_16(weights=weights)
        hidden = network.heads.head.in_features
        network.heads.head = nn.Identity()
        self.backbone = network
        self.head = nn.Linear(hidden, n_classes)
        self.n_classes = n_classes
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def expand_head(self, n_classes: int) -> None:
        if n_classes <= self.n_classes:
            return
        self.head = _grow_linear_head(self.head, n_classes)
        self.n_classes = n_classes
