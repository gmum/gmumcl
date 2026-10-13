"""Shared test helpers for the AdaReg core unit tests."""

from __future__ import annotations

from typing import Tuple

import torch


def random_region(dimension: int = 12, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Well-separated random (d, w, q) region in float64."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.cumsum(torch.rand(dimension, generator=generator, dtype=torch.float64) + 0.3, dim=0)
    d = torch.rand(dimension, generator=generator, dtype=torch.float64) + 0.5
    w = 0.5 * torch.rand(dimension, generator=generator, dtype=torch.float64)
    return d, w, q


def stressed_region(dimension: int = 64, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adversarial region: minimal gaps, wide spread and near-singular d."""
    generator = torch.Generator().manual_seed(seed)
    gaps = 1e-3 + 3.0 * torch.rand(dimension - 1, generator=generator, dtype=torch.float64)
    q = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(gaps, dim=0)])
    d = 10.0 ** (-6.0 * torch.rand(dimension, generator=generator, dtype=torch.float64))
    w = torch.rand(dimension, generator=generator, dtype=torch.float64)
    return d, w, q


def dense_shape(d: torch.Tensor, w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    kernel = torch.exp(-torch.abs(q[:, None] - q[None, :]))
    return torch.diag(d) + (w[:, None] * w[None, :]) * kernel


def dense_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * torch.sqrt(values.clamp_min(0.0))) @ vectors.transpose(0, 1)


def containing_region(
    points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """A diagonal region of radius 1 containing all rows of ``points``.

    With S = diag(d) and d_k = D * max_n (z_nk - mu_k)^2 the Mahalanobis
    distance of every point is at most 1.
    """
    dimension = points.shape[-1]
    mu = points.mean(dim=0)
    deviation = (points - mu).abs().amax(dim=0)
    d = (dimension * deviation * deviation).clamp_min(1e-12)
    q = torch.arange(dimension, dtype=points.dtype, device=points.device)
    w = torch.zeros_like(d)
    return mu, d, w, q, 1.0
