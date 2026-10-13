"""Parameter-tensor helpers for a classification head that grows per task.

With ``model.growing_head: true`` the head starts at the first task's classes
and the trainer appends rows for each next task (`expand_head` keeps the old
rows). Every old-task statistic (snapshot, importance, Kronecker factor,
interval) predates those rows, so old-task quantities use the leading rows and
the rows added for the current task stay unregularised until its task end
(D18).
"""

from __future__ import annotations

import torch


def aligned_rows(parameter: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Leading rows of ``parameter``, trimmed to a smaller ``reference``."""
    if parameter.shape == reference.shape:
        return parameter
    return parameter[: reference.shape[0]]


def padded_rows(reference: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    """A smaller ``reference`` zero-padded to the shape of ``parameter``."""
    if reference.shape == parameter.shape:
        return reference
    padded = torch.zeros_like(parameter)
    padded[: reference.shape[0]] = reference
    return padded


def padded_square(matrix: torch.Tensor, rows: int) -> torch.Tensor:
    """A square factor zero-padded to ``rows`` x ``rows``."""
    if matrix.shape[0] == rows:
        return matrix
    padded = matrix.new_zeros(rows, rows)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return padded
