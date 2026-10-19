"""Unit tests for the model placement helpers."""

import pytest
import torch
import torch.nn as nn

from gmumcl.cl.models import (
    MLP,
    ResNet18,
    ViT,
    paired_batchnorm,
    placement_include,
    protected_layers,
)


def test_paired_batchnorm_finds_the_following_sibling():
    model = ResNet18(n_classes=4)
    assert isinstance(paired_batchnorm(model, "backbone.conv1"), nn.BatchNorm2d)
    assert isinstance(paired_batchnorm(model, "backbone.layer1.0.conv1"), nn.BatchNorm2d)


def test_placement_patterns_select_the_resnet_head_and_last_blocks():
    model = ResNet18(n_classes=4)
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
    model = ViT(n_classes=4, freeze_backbone=True)
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
        placement_include("nope", ResNet18(n_classes=4))

    class Headless(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(4, 4)

        def forward(self, x):
            return self.features(x)

    with pytest.raises(ValueError):
        protected_layers(Headless(), "*", "head")


def test_mlp_protected_layers_are_ordered_by_name():
    model = MLP(input_dim=4, n_classes=4, hidden=8, layers=2)
    names = [name for name, _ in protected_layers(model, "*head", "head")]
    assert names == ["head"]
    assert torch.equal(model.head.weight, model.get_submodule("head").weight)
