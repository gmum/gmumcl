"""Shared baseline utilities: snapshots, diagonal Fisher, Kronecker factors."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..growth import aligned_rows

Factor = Dict[str, torch.Tensor]


def parameter_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Detached on-device copies of every parameter, keyed by name.

    Used by baselines that need no activation capture: unlike
    ``ActivationScope.capture_parameters`` it needs no tracker instance and
    keeps tensors on the parameter device (deltas are read every step).
    """
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def diagonal_fisher(
    model: nn.Module,
    loader,
    device: str,
    flip_logits: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Diagonal Fisher of log p(y|x), squared-gradient accumulation."""
    fisher = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
    model.eval()
    batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        logits = model(x)
        if flip_logits:
            logits = -logits
        loss = F.cross_entropy(logits, y)
        gradients = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
        for (name, parameter), gradient in zip(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad], gradients
        ):
            fisher[name] += gradient.detach() ** 2
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    model.zero_grad()
    return fisher


def fisher_quadratic(
    model: nn.Module, snapshot: Dict[str, torch.Tensor], fisher: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """sum_fisher * (theta - snapshot)^2, a scalar.

    A grown head contributes only the rows captured in ``snapshot`` (D18).
    """
    total = None
    for name, parameter in model.named_parameters():
        difference = aligned_rows(parameter, snapshot[name]) - snapshot[name]
        term = (fisher[name] * difference**2).sum()
        total = term if total is None else total + term
    return total if total is not None else torch.zeros(())


def input_second_moment(module: nn.Module, activation: torch.Tensor) -> torch.Tensor:
    """E[a_hat a_hat^T] with the bias coordinate appended (Kronecker A factor)."""
    if isinstance(module, nn.Conv2d):
        patches = torch.nn.functional.unfold(
            activation, module.kernel_size, stride=module.stride, padding=module.padding
        ).transpose(1, 2)
        flat = patches.reshape(-1, patches.shape[-1])
    else:
        flat = activation.reshape(activation.shape[0], -1)
    augmented = torch.cat(
        [flat, torch.ones(flat.shape[0], 1, dtype=flat.dtype, device=flat.device)], dim=1
    ).double()
    return augmented.transpose(0, 1) @ augmented / augmented.shape[0]


def _sampled_targets(logits: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """softmax_sampling from the official K-FAC code: a target drawn from the model."""
    probabilities = torch.softmax(logits.detach(), dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def output_gradient_covariance(
    model: nn.Module,
    loader,
    layer_names,
    device: str,
    max_samples: int = 256,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """E[g g^T] per layer for g = d log p / d activation at sampled targets.

    The output capture keeps the autograd graph because ``g`` is differentiated
    through it; ActivationScope snapshots are detached and cannot be used here.
    """
    outputs: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if name in layer_names:
            handles.append(
                module.register_forward_hook(
                    lambda mod, inp, out, n=name: outputs.__setitem__(n, out)
                )
            )
    accumulation: Dict[str, torch.Tensor] = {}
    generator_device = "cuda" if torch.device(device).type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    count = 0
    model.eval()
    try:
        for x, _ in loader:
            for i in range(x.shape[0]):
                if count >= max_samples:
                    break
                model.zero_grad()
                logits = model(x[i : i + 1].to(device))
                target = _sampled_targets(logits, generator)
                log_probability = torch.log_softmax(logits, dim=-1)[0, target].sum()
                for name in layer_names:
                    activation = outputs[name]
                    gradient = torch.autograd.grad(
                        log_probability, activation, retain_graph=True
                    )[0]
                    if gradient.dim() == 4:
                        gradient = gradient.mean(dim=(0, 2, 3))
                    else:
                        gradient = gradient[0]
                    vector = gradient.reshape(-1).double()
                    outer = torch.outer(vector, vector)
                    accumulation[name] = accumulation.get(name, torch.zeros_like(outer)) + outer
                count += 1
            if count >= max_samples:
                break
    finally:
        for handle in handles:
            handle.remove()
    return {name: value / max(count, 1) for name, value in accumulation.items()}


# Kronecker sums and regularised inverses (NCL prior merge and preconditioner).
# Port of `tachukao/ncl` (commit 92c2ba5e3f20c42dff3b9e2e7d93c8d6d49077a3),
# `ncl/learners/fisher/kfsum.py` and `ncl/learners/tools.py` (`reginv`).

_EARLY_STOP = 1e-7
_GOOD_ENOUGH = 5e-4


def recondition(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetric positive semi-definite part (negative eigenvalues clipped)."""
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(0.0)
    return (vectors * values.unsqueeze(0)) @ vectors.T


def safe_inv(matrix: torch.Tensor, jitter: float = 1e-8) -> torch.Tensor:
    """Trace-normalised inverse with eigenvalues clipped at ``jitter``."""
    matrix = 0.5 * (matrix + matrix.T)
    dimension = matrix.shape[0]
    scale = dimension / torch.trace(matrix)
    values, vectors = torch.linalg.eigh(scale * matrix)
    values = values.clamp_min(jitter)
    return scale * (vectors * (1.0 / values).unsqueeze(0)) @ vectors.T


def additive_nearest_kf(prior: Factor, update: Factor) -> Factor:
    """Scaled additive approximation with the trace-optimal ``pi`` (Eq. 43/48)."""
    left, right = prior["left"], prior["right"]
    new_left, new_right = update["left"], update["right"]
    pi = torch.sqrt(torch.trace(left) * torch.trace(new_right)) / torch.sqrt(
        torch.trace(new_left) * torch.trace(right)
    )
    return {"left": left + new_left * pi, "right": right + new_right / pi}


def _balance(factor: Factor) -> Factor:
    """Rescale ``left``/``right`` to a common trace-per-dimension scale."""
    left, right = factor["left"], factor["right"]
    scale = (torch.trace(left) * torch.trace(right)) / (right.shape[0] * left.shape[0])
    root = torch.sqrt(scale)
    return {
        "left": root * left / (torch.trace(left) / left.shape[0]),
        "right": root * right / (torch.trace(right) / right.shape[0]),
    }


def _kf_step(
    left: torch.Tensor,
    right: torch.Tensor,
    prior_left: torch.Tensor,
    prior_right: torch.Tensor,
    new_left: torch.Tensor,
    new_right: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One KL fixed-point update of the Kronecker sum (Eq. 60/61)."""
    dimension_left = prior_left.shape[0]
    dimension_right = prior_right.shape[0]
    left = recondition(left)
    right = recondition(right)
    left_inv = safe_inv(left)
    right_inv = safe_inv(right)
    e1 = torch.trace(prior_right @ right_inv) / dimension_right
    e2 = torch.trace(new_right @ right_inv) / dimension_right
    f1 = torch.trace(prior_left @ left_inv) / dimension_left
    f2 = torch.trace(new_left @ left_inv) / dimension_left
    difference_right = f1 * prior_right + f2 * new_right - right
    difference_left = e1 * prior_left + e2 * new_left - left
    return (
        left + beta * difference_left,
        right + beta * difference_right,
        torch.mean(difference_left**2) / torch.mean(left**2),
        torch.mean(difference_right**2) / torch.mean(right**2),
    )


def opt_nearest_kf(
    prior: Factor, update: Factor, max_steps: int = 1000, beta: float = 0.3
) -> Factor:
    """KL-optimal Kronecker approximation of ``prior + update`` (Appendix G)."""
    prior = _balance(prior)
    update = _balance(update)
    prior_left = recondition(prior["left"])
    prior_right = recondition(prior["right"])
    new_left = recondition(update["left"])
    new_right = recondition(update["right"])
    factor = additive_nearest_kf(
        {"left": prior_left, "right": prior_right},
        {"left": new_left, "right": new_right},
    )
    left, right = factor["left"], factor["right"]
    delta_left, delta_right = float("inf"), float("inf")
    for _ in range(max_steps):
        left, right, delta_left, delta_right = _kf_step(
            left, right, prior_left, prior_right, new_left, new_right, beta
        )
        if (delta_left < _EARLY_STOP and delta_right < _EARLY_STOP) or (
            delta_left < _GOOD_ENOUGH and delta_right < _GOOD_ENOUGH
        ):
            return {
                "left": 0.5 * (left + left.T),
                "right": 0.5 * (right + right.T),
            }
    raise RuntimeError(
        "nearest-KF sum did not converge in %d steps (dL=%.2e, dR=%.2e)"
        % (max_steps, delta_left, delta_right)
    )


def reginv(matrix: torch.Tensor, alpha: float) -> torch.Tensor:
    """Regularised symmetric inverse ``(X + alpha I)^-1`` (``tools.reginv``)."""
    matrix = 0.5 * (matrix + matrix.T)
    scale = torch.trace(matrix) / matrix.shape[0]
    values, vectors = torch.linalg.eigh(matrix / scale)
    values = 1.0 / (alpha + values * scale)
    return (vectors * values.unsqueeze(0)) @ vectors.T
