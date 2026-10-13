"""Unit tests for the RBF head and the Dreaming plugin wiring."""

import math

import pytest
import torch

from gmumcl.cl.baselines import build as build_baseline
from gmumcl.cl.models import MLP
from gmumcl.cl.rbf import RBFHead
from gmumcl.cl.scenarios import synthetic_tasks
from gmumcl.cl.trainer import Trainer, TrainerConfig


def test_rbf_head_output_shape_and_gradients():
    head = RBFHead(in_features=5, n_classes=3)
    x = torch.randn(7, 5, requires_grad=True)
    output = head(x)
    assert output.shape == (7, 3)
    assert torch.all(output >= 0) and torch.all(output <= 1)
    output.sum().backward()
    assert x.grad is not None


def test_rbf_head_expand_keeps_the_old_rows():
    torch.manual_seed(0)
    head = RBFHead(in_features=4, n_classes=2)
    x = torch.randn(3, 4)
    before = head(x)
    centers = head.kernels_centers.detach().clone()
    head.expand_head(5)
    after = head(x)
    assert head.n_classes == 5
    assert torch.allclose(before, after[:, :2])
    assert torch.equal(head.kernels_centers[:2], centers)


def test_dreaming_requires_an_rbf_head():
    plugin = build_baseline("dreaming")
    with pytest.raises(TypeError, match="RBF head"):
        plugin.set_model(MLP(input_dim=4, n_classes=2, hidden=8))


def test_dreaming_generates_and_trains_on_two_tasks():
    tasks = synthetic_tasks(
        num_classes=4,
        classes_per_task=2,
        dimension=4,
        samples_per_class=20,
        min_calibration=4,
        batch_size=8,
        seed=0,
    )
    model = MLP(input_dim=4, n_classes=2, hidden=8, head_type="rbf")
    plugin = build_baseline(
        "dreaming",
        {"alpha": 0.5, "samples_per_class": 3, "steps": 2, "in_shape": [4]},
    )
    trainer = Trainer(model, [plugin], TrainerConfig(epochs=1, lr=0.01))
    matrix = trainer.run(tasks[:2])
    assert plugin.dreamed, "second task should hold dreamed inputs per old class"
    assert len(plugin.dreamed) == 2
    assert all(math.isfinite(value) for row in matrix.rows for value in row)
