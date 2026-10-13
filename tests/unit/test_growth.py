"""Unit tests for the growing-head helpers (D18)."""

import torch

from gmumcl.cl.baselines.common import fisher_quadratic, parameter_snapshot
from gmumcl.cl.baselines.penalties import L2
from gmumcl.cl.growth import aligned_rows, padded_rows, padded_square
from gmumcl.cl.models import MLP


def test_aligned_rows_trims_a_grown_parameter():
    reference = torch.randn(2, 3)
    grown = torch.randn(5, 3)
    assert aligned_rows(grown, reference).shape == (2, 3)
    assert aligned_rows(reference, reference) is reference


def test_padded_rows_zero_fills_the_new_rows():
    reference = torch.ones(2, 3)
    parameter = torch.zeros(5, 3)
    padded = padded_rows(reference, parameter)
    assert padded.shape == (5, 3)
    assert torch.equal(padded[:2], reference)
    assert padded[2:].abs().sum() == 0


def test_padded_square_zero_fills_both_dimensions():
    matrix = torch.ones(2, 2)
    padded = padded_square(matrix, 4)
    assert padded.shape == (4, 4)
    assert torch.equal(padded[:2, :2], matrix)
    assert padded[2:].abs().sum() == 0
    assert padded[:, 2:].abs().sum() == 0


def test_expand_head_preserves_the_old_outputs():
    torch.manual_seed(0)
    model = MLP(input_dim=6, n_classes=2, hidden=8)
    x = torch.randn(4, 6)
    before = model(x)
    model.expand_head(6)
    after = model(x)
    assert model.n_classes == 6
    assert torch.allclose(before, after[:, :2])


def test_expand_head_keeps_device_and_dtype():
    model = MLP(input_dim=6, n_classes=2, hidden=8).double()
    model.expand_head(4)
    assert model.head.weight.dtype == torch.float64
    assert model.head.bias.dtype == torch.float64


def test_fisher_quadratic_ignores_rows_added_after_the_snapshot():
    model = MLP(input_dim=6, n_classes=2, hidden=8)
    snapshot = parameter_snapshot(model)
    fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    model.expand_head(4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    expected = sum((value**2).sum() for value in snapshot.values())
    assert torch.isclose(fisher_quadratic(model, snapshot, fisher), expected)


def test_l2_penalty_ignores_rows_added_after_the_snapshot():
    model = MLP(input_dim=6, n_classes=2, hidden=8)
    plugin = L2({"lambda": 1.0})
    plugin.set_model(model)
    plugin.on_task_end(0, None)
    model.expand_head(4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    composed, _ = plugin.forward(None, None, torch.zeros(()), None)
    expected = sum((value**2).sum() for value in plugin.snapshot.values())
    assert torch.isclose(composed, expected)
