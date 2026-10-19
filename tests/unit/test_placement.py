"""Unit tests for the model placement helpers.

The stubs reproduce the torchvision module naming (``backbone.layer4``,
``encoder.layers.encoder_layer_11``, paired BatchNorms) so the helpers are
tested without importing torchvision.
"""

import pytest
import torch
import torch.nn as nn

from gmumcl.cl.models import (
    MLP,
    paired_batchnorm,
    placement_include,
    protected_layers,
)


class _TinyBlock(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(x + self.bn2(self.conv2(hidden)))


class _TinyResNet(nn.Module):
    """ResNet-18-like naming: conv1/BN pairs, layer1..layer4, linear head."""

    def __init__(self, n_classes: int = 4, channels: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = nn.Module()
        self.backbone.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.backbone.bn1 = nn.BatchNorm2d(channels)
        for name in ("layer1", "layer2", "layer3", "layer4"):
            setattr(
                self.backbone,
                name,
                nn.ModuleList([_TinyBlock(channels), _TinyBlock(channels)]),
            )
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.backbone.bn1(self.backbone.conv1(x)))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            for block in getattr(self.backbone, name):
                hidden = block(hidden)
        return self.head(hidden.flatten(1))

    def expand_head(self, n_classes: int) -> None:
        return


class _TinyViT(nn.Module):
    """ViT-B/16-like naming: encoder.layers.encoder_layer_11.mlp, wrapper head."""

    def __init__(self, n_classes: int = 4, width: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = nn.Module()
        encoder = nn.Module()
        encoder.layers = nn.Module()
        for index in range(12):
            layer = nn.Module()
            layer.mlp = nn.Sequential(
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.Linear(width, width),
            )
            setattr(encoder.layers, "encoder_layer_%d" % index, layer)
        self.backbone.encoder = encoder
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def expand_head(self, n_classes: int) -> None:
        return


def test_paired_batchnorm_finds_the_following_sibling():
    model = _TinyResNet()
    assert isinstance(paired_batchnorm(model, "backbone.conv1"), nn.BatchNorm2d)
    assert isinstance(paired_batchnorm(model, "backbone.layer1.0.conv1"), nn.BatchNorm2d)
    assert paired_batchnorm(model, "backbone.layer1.0.bn1") is None


def test_placement_patterns_select_the_resnet_head_and_last_blocks():
    model = _TinyResNet()
    names = [name for name, _ in protected_layers(model, placement_include("head", model), "head")]
    assert names == ["head"]
    last_block = [
        name
        for name, _ in protected_layers(
            model, placement_include("head_and_last_blocks", model), "head"
        )
    ]
    assert "backbone.layer4.0.conv1" in last_block
    assert last_block[-1] == "head"


def test_vit_placement_patterns_select_the_wrapper_head():
    model = _TinyViT()
    names = [
        name
        for name, _ in protected_layers(model, placement_include("vit_head", model), "head")
    ]
    assert names == ["head"]
    last_block = [
        name
        for name, _ in protected_layers(
            model, placement_include("vit_head_and_last_block", model), "head"
        )
    ]
    assert "backbone.encoder.layers.encoder_layer_11.mlp.3" in last_block
    assert last_block[-1] == "head"


def test_unknown_placement_and_missing_head_raise():
    with pytest.raises(KeyError):
        placement_include("nope", _TinyResNet())

    with pytest.raises(ValueError):
        protected_layers(_TinyResNet().backbone, "*conv1", "head")


def test_mlp_protected_layers_are_ordered_by_name():
    model = MLP(input_dim=4, n_classes=4, hidden=8, layers=2)
    names = [name for name, _ in protected_layers(model, "*head", "head")]
    assert names == ["head"]
    assert torch.equal(model.head.weight, model.get_submodule("head").weight)
