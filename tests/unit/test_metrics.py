"""Unit tests for task_accuracy and the task-oracle masking."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gmumcl.cl.metrics import AccuracyMatrix, task_accuracy


class _FixedLogits(nn.Module):
    """Always returns the same four-class logits for every sample."""

    def forward(self, x):
        logits = torch.tensor([[5.0, 4.0, 100.0, 0.0]])
        return logits.repeat(x.shape[0], 1)


def _loader():
    x = torch.zeros(4, 3)
    y = torch.tensor([0, 0, 1, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_task_accuracy_uses_the_full_head_without_oracle():
    accuracy = task_accuracy(_FixedLogits(), _loader(), "cpu", classes=[0, 1])
    assert accuracy == 0.0, "class 2 dominates the full head"


def test_task_accuracy_masks_logits_with_oracle():
    accuracy = task_accuracy(
        _FixedLogits(), _loader(), "cpu", classes=[0, 1], mask_logits=True
    )
    assert accuracy == 0.5, "within classes [0, 1] class 0 wins"


def test_task_accuracy_ignores_other_classes():
    x = torch.zeros(4, 3)
    y = torch.tensor([0, 0, 1, 2])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    accuracy = task_accuracy(_FixedLogits(), loader, "cpu", classes=[0, 1])
    assert accuracy == 0.0  # only the class-0/1 samples are scored, class 2 wins


def test_accuracy_matrix_metrics():
    matrix = AccuracyMatrix(rows=[[0.9], [0.6, 0.8], [0.5, 0.7, 0.9]])
    assert abs(matrix.old_accuracy(2) - 0.6) < 1e-9
    assert abs(matrix.new_accuracy(2) - 0.9) < 1e-9
    assert abs(matrix.forgetting() - (0.4 + 0.1) / 2) < 1e-9
    assert 0.0 <= matrix.average_incremental_accuracy() <= 1.0
