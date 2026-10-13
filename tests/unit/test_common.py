"""Unit tests for the NCL Kronecker-sum helpers, against dense references."""

import torch

from gmumcl.cl.baselines.common import (
    additive_nearest_kf,
    opt_nearest_kf,
    reginv,
    safe_inv,
)


def _spd(dimension, scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    return matrix @ matrix.T + scale * torch.eye(dimension, dtype=torch.float64)


def _dense(factor):
    return torch.kron(factor["left"], factor["right"])


def _gaussian_kl(left, right):
    """KL(N(0, left) || N(0, right)) for the dense precision matrices."""
    dimension = left.shape[0]
    jitter = 1e-9 * torch.eye(dimension, dtype=torch.float64)
    left = 0.5 * (left + left.T) + jitter
    right = 0.5 * (right + right.T) + jitter
    _, log_left = torch.linalg.slogdet(left)
    _, log_right = torch.linalg.slogdet(right)
    solved = torch.linalg.solve(right, left)
    return 0.5 * (torch.trace(solved) - dimension + log_right - log_left)


def test_reginv_matches_the_dense_inverse():
    matrix = _spd(5, scale=2.0, seed=1)
    expected = torch.linalg.inv(matrix + 1e-6 * torch.eye(5, dtype=torch.float64))
    assert torch.allclose(reginv(matrix, 1e-6), expected, atol=1e-8)


def test_safe_inv_matches_the_dense_inverse():
    matrix = _spd(4, scale=3.0, seed=2)
    assert torch.allclose(safe_inv(matrix), torch.linalg.inv(matrix), atol=1e-5)


def test_additive_nearest_kf_uses_the_trace_optimal_pi():
    prior = {"left": _spd(4, 2.0, 3), "right": _spd(3, 2.0, 4)}
    update = {"left": _spd(4, 1.0, 5), "right": _spd(3, 1.0, 6)}
    pi = torch.sqrt(torch.trace(prior["left"]) * torch.trace(update["right"])) / torch.sqrt(
        torch.trace(update["left"]) * torch.trace(prior["right"])
    )
    merged = additive_nearest_kf(prior, update)
    assert torch.allclose(merged["left"], prior["left"] + pi * update["left"])
    assert torch.allclose(merged["right"], prior["right"] + update["right"] / pi)


def test_opt_nearest_kf_beats_the_additive_approximation():
    prior = {"left": _spd(4, 2.0, 7), "right": _spd(3, 2.0, 8)}
    update = {"left": _spd(4, 1.0, 9), "right": _spd(3, 1.0, 10)}
    target = _dense(prior) + _dense(update)
    additive = _dense(additive_nearest_kf(prior, update))
    optimal = _dense(opt_nearest_kf(prior, update, max_steps=500))
    assert _gaussian_kl(optimal, target) < _gaussian_kl(additive, target)


def test_opt_nearest_kf_recovers_an_isotropic_sum():
    eye_left = torch.eye(4, dtype=torch.float64)
    eye_right = torch.eye(3, dtype=torch.float64)
    prior = {"left": 2.0 * eye_left, "right": 2.0 * eye_right}
    update = {"left": 3.0 * eye_left, "right": 3.0 * eye_right}
    target = 13.0 * torch.kron(eye_left, eye_right)
    optimal = _dense(opt_nearest_kf(prior, update, max_steps=500))
    # the official early stop (dL, dR < 5e-4) leaves sub-percent accuracy here
    assert _gaussian_kl(optimal, target) < 1e-3
