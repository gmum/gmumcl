"""Projection baselines: OWM and GPM, ported from the official code.

- OWM: `GMvandeVen/continual-learning` (commit
  e6d795aa81b9cef742b8de76cb71222d4d1ce00b), the base of the NCL paper's
  feedforward fork (`estimate_owm_fisher`). After each task the augmented input
  second moment `A = E[[a, 1][a, 1]^T]` of every affine layer is estimated over
  the task data and averaged across tasks; during the next task the augmented
  gradient `[g_W | g_b]` is projected with `P = I - A (A + alpha I)^-1`, which
  leaves old pre-activations (and so old logits) unchanged up to the stabiliser
  `alpha` (Table 1 CIL optimum 1e-4). The bias is projected with the weights.
- GPM: `sahagobinda/GPM`
  (commit 1a238ec9d2ca30bae8fd1707f161cc6bd093c72a), `main_cifar100.py`.
  Task-end representation matrices are SVD'd with the energy thresholds
  `0.97 + 0.003 * task` (first task: cumulative energy; later tasks: residual
  energy after projecting on the stored basis, Eq. 5/8/9 of the paper), and
  weight gradients are projected as `g <- g - (g U) U^T`. Biases of protected
  layers are frozen after the first task, as in the official code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..collect import collect
from ..plugin import MethodPluginABC
from .common import input_second_moment


def _affine_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]


def _representation_matrix(module: nn.Module, activation: torch.Tensor, max_rows: int) -> torch.Tensor:
    """Rows = examples/patches, columns = input dimension."""
    activation = activation.double()
    if isinstance(module, nn.Conv2d):
        patches = torch.nn.functional.unfold(
            activation, module.kernel_size, stride=module.stride, padding=module.padding
        ).transpose(1, 2)
        flat = patches.reshape(-1, patches.shape[-1])
    else:
        flat = activation.reshape(activation.shape[0], -1)
    if flat.shape[0] > max_rows:
        index = torch.randperm(flat.shape[0])[:max_rows]
        flat = flat[index]
    return flat.transpose(0, 1)  # (input_dim, rows), as in the official code


class OWM(MethodPluginABC):
    """Orthogonal Weight Modification with the official augmented projector."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.snapshots: Dict[str, torch.Tensor] = {}
        self.projections: Dict[str, torch.Tensor] = {}
        self.tasks_seen = 0

    def on_after_backward(self) -> None:
        if not self.projections or self.model is None:
            return
        for name, module in _affine_layers(self.model):
            projector = self.projections.get(name)
            if projector is None or module.weight.grad is None:
                continue
            weight_gradient = module.weight.grad.reshape(module.weight.shape[0], -1)
            if module.bias is not None and module.bias.grad is not None:
                augmented = torch.cat(
                    [weight_gradient, module.bias.grad.unsqueeze(1)], dim=1
                )
            else:
                augmented = torch.cat(
                    [weight_gradient, weight_gradient.new_zeros(weight_gradient.shape[0], 1)],
                    dim=1,
                )
            projected = augmented.double() @ projector.double()
            module.weight.grad.copy_(
                projected[:, : weight_gradient.shape[1]].reshape_as(module.weight)
            )
            if module.bias is not None and module.bias.grad is not None:
                module.bias.grad.copy_(projected[:, -1])

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        device = next(self.model.parameters()).device
        layers = _affine_layers(self.model)
        names = [name for name, _ in layers]
        activations = collect(self.model, names, task.fit_loader, device, "input")
        alpha = float(self.config.get("alpha", 1e-10))
        weight = 1.0 / (self.tasks_seen + 1)
        for name, module in layers:
            second_moment = input_second_moment(module, activations[name])
            previous = self.snapshots.get(name)
            merged = (
                second_moment
                if previous is None
                else (1.0 - weight) * previous + weight * second_moment
            )
            self.snapshots[name] = merged
            stabilised = merged + alpha * torch.eye(merged.shape[0], dtype=merged.dtype)
            projector = torch.eye(merged.shape[0], dtype=merged.dtype) - (
                merged @ torch.linalg.inv(stabilised)
            )
            self.projections[name] = projector.to(
                device=module.weight.device, dtype=module.weight.dtype
            )
        self.tasks_seen += 1


class GPM(MethodPluginABC):
    """Gradient Projection Memory with the official energy thresholds."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.bases: Dict[str, torch.Tensor] = {}

    def setup_task(self, task_id: int, task: Optional[Any] = None) -> None:
        self.task_id = task_id

    def on_after_backward(self) -> None:
        if not self.bases or self.model is None:
            return
        for name, module in _affine_layers(self.model):
            basis = self.bases.get(name)
            if basis is None or module.weight.grad is None:
                continue
            gradient = module.weight.grad.reshape(module.weight.shape[0], -1)
            basis = basis.to(device=gradient.device, dtype=gradient.dtype)
            gradient = gradient - (gradient @ basis) @ basis.transpose(0, 1)
            module.weight.grad.copy_(gradient.reshape_as(module.weight))
            if module.bias is not None and module.bias.grad is not None and self.task_id > 0:
                module.bias.grad.zero_()  # official code freezes biases after task 0

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        device = next(self.model.parameters()).device
        layers = _affine_layers(self.model)
        names = [name for name, _ in layers]
        activations = collect(self.model, names, task.fit_loader, device, capture="input")
        threshold = float(self.config.get("threshold_base", 0.97)) + float(
            self.config.get("threshold_step", 0.003)
        ) * task_id
        max_rows = int(self.config.get("max_rows", 125))
        for name, module in layers:
            matrix = _representation_matrix(module, activations[name], max_rows)
            if name not in self.bases:
                left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
                energy = singular**2
                cumulative = torch.cumsum(energy / energy.sum(), dim=0)
                rank = int((cumulative < threshold).sum().item())
                self.bases[name] = left[:, : max(rank, 1)].float()
                continue
            basis = self.bases[name].double()
            _, singular_all, _ = torch.linalg.svd(matrix, full_matrices=False)
            total = (singular_all**2).sum()
            residual = matrix - basis @ (basis.transpose(0, 1) @ matrix)
            left, singular, _ = torch.linalg.svd(residual, full_matrices=False)
            accumulated = (total - (singular**2).sum()) / total
            rank = 0
            for ii in range(singular.shape[0]):
                if accumulated < threshold:
                    accumulated = accumulated + (singular[ii] ** 2) / total
                    rank += 1
                else:
                    break
            if rank == 0:
                continue
            combined = torch.cat([basis, left[:, :rank]], dim=1)
            if combined.shape[1] > combined.shape[0]:
                combined = combined[:, : combined.shape[0]]
            self.bases[name] = combined.float()
