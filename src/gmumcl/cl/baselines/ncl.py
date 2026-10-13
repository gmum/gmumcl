"""Natural Continual Learning (Kao et al., NeurIPS 2021).

Port of `tachukao/ncl` (commit 92c2ba5e3f20c42dff3b9e2e7d93c8d6d49077a3):
`ncl/learners/laplace.py` (classes ``KFAC`` and ``NCL``, ``update_priors``),
`ncl/learners/fisher/kfsum.py` (KL-optimal Kronecker sums) and
`ncl/learners/tools.py` (``reginv``), for feed-forward blocks:

- per-layer factors A (augmented input second moment) and G (output-gradient
  covariance), as in our K-FAC baseline,
- the prior starts at ``p_w I`` per factor with
  ``p_w = data_size ** -0.5`` (``data_size`` is the paper's ``p_w^{-2}``);
  after each task it is merged with the task factors by the additive
  nearest-Kronecker sum of the feedforward codebase
  (`GMvandeVen/continual-learning`, `models/utils/ncl.py`); the paper's
  KL-optimal merge (Appendix G) is numerically unstable over long runs, so
  the additive variant is used (documented deviation),
- the penalty is ``0.5 * sum(A (W - W_prev) G) * (W - W_prev)`` with the raw
  prior factors and, for the first task, ``W_prev = 0``,
- each task's projections are ``reginv`` of the prior factors after adding the
  stabiliser ``alpha I``; ``on_after_backward`` replaces the gradient by
  ``P_right g P_left``, so with the penalty included the SGD momentum update is
  the paper's ``theta <- theta + gamma [Lambda^-1 grad(l) - (theta - mu)]``
  (Eq. 8). The transformed gradients are clipped to norm 10 for stability.
  The ``data_size`` config key only sets the prior scale
  ``p_w = data_size ** -0.5``; the RNN reference code additionally divides its
  step size by ``data_size`` with ``data_size = 1``, which the feedforward
  paper does not do.

Documented deviations from the upstream code:

- the official code is recurrent; this is the per-layer feed-forward port that
  shares the K-FAC factors (convolution factors averaged over locations),
- the classification head grows per task (D18) instead of being pre-allocated:
  output-side prior rows for classes added during the current task are zero, so
  those rows stay unregularised until the task ends,
- the momentum state is re-created at each task switch, while the official
  learner carries it across tasks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..collect import collect
from ..growth import padded_rows, padded_square
from ..plugin import MethodPluginABC
from .common import (
    additive_nearest_kf,
    input_second_moment,
    output_gradient_covariance,
    reginv,
)


class NCL(MethodPluginABC):
    """Official NCL with per-layer Kronecker priors and preconditioned updates."""

    def _layers(self) -> List[Tuple[str, nn.Module]]:
        assert self.model is not None
        return [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]

    @staticmethod
    def _input_dim(module: nn.Module) -> int:
        return module.weight[0].numel() + 1

    def _init_priors(self) -> None:
        p_w = float(self.config.get("data_size", 1.0)) ** -0.5
        self.left: Dict[str, torch.Tensor] = {}
        self.right: Dict[str, torch.Tensor] = {}
        for name, module in self._layers():
            self.left[name] = torch.eye(self._input_dim(module), dtype=torch.float64) * p_w
            self.right[name] = torch.eye(module.weight.shape[0], dtype=torch.float64) * p_w

    def _build_projections(self) -> None:
        assert self.model is not None
        alpha = float(self.config.get("projection_alpha", 1e-10))
        self.proj_left: Dict[str, torch.Tensor] = {}
        self.proj_right: Dict[str, torch.Tensor] = {}
        for name, module in self._layers():
            left, right = self.left[name], self.right[name]
            merged = additive_nearest_kf(
                {"left": left, "right": right},
                {
                    "left": alpha * torch.eye(left.shape[0], dtype=torch.float64),
                    "right": alpha * torch.eye(right.shape[0], dtype=torch.float64),
                },
            )
            self.proj_left[name] = reginv(merged["left"], 1e-4).to(module.weight.device).float()
            self.proj_right[name] = reginv(merged["right"], 1e-4).to(module.weight.device).float()

    def setup_task(self, task_id: int) -> None:
        assert self.model is not None
        self.task_id = task_id
        if not hasattr(self, "left"):
            self._init_priors()
        else:
            for name, module in self._layers():
                self.right[name] = padded_square(self.right[name], module.weight.shape[0])
        self.anchor = None
        if task_id > 0:
            self.anchor = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
            }
        self._build_projections()

    def _augmented_delta(self, name: str, module: nn.Module) -> torch.Tensor:
        weight = module.weight.reshape(module.weight.shape[0], -1).double()
        if module.bias is not None:
            bias = module.bias.double()
        else:
            bias = weight.new_zeros(weight.shape[0])
        if getattr(self, "anchor", None) is None:
            return torch.cat([weight, bias.unsqueeze(1)], dim=1)
        anchor_weight = (
            padded_rows(self.anchor[name + ".weight"], module.weight)
            .reshape(module.weight.shape[0], -1)
            .double()
        )
        anchor_bias = padded_rows(self.anchor[name + ".bias"], module.bias).double()
        return torch.cat([weight - anchor_weight, (bias - anchor_bias).unsqueeze(1)], dim=1)

    def on_after_backward(self) -> None:
        if self.model is None or not hasattr(self, "proj_left"):
            return
        for name, module in self._layers():
            if module.weight.grad is None or name not in self.proj_left:
                continue
            gradient = module.weight.grad
            rows = gradient.shape[0]
            if module.bias is not None and module.bias.grad is not None:
                augmented = torch.cat(
                    [gradient.reshape(rows, -1), module.bias.grad.unsqueeze(1)], dim=1
                )
            else:
                augmented = torch.cat(
                    [gradient.reshape(rows, -1), gradient.new_zeros(rows, 1)], dim=1
                )
            projected = self.proj_right[name] @ augmented @ self.proj_left[name]
            limit = float(self.config.get("grad_clip", 10.0))
            norm = projected.norm()
            if norm > limit:
                projected = projected * (limit / norm)
            gradient.copy_(projected[:, :-1].reshape_as(gradient))
            if module.bias is not None and module.bias.grad is not None:
                module.bias.grad.copy_(projected[:, -1])

    def forward(self, x, y, loss, logits):
        if not hasattr(self, "left") or not self.left:
            return loss, logits
        total = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for name, module in self._layers():
            if name not in self.left:
                continue
            delta = self._augmented_delta(name, module)
            left = self.left[name].to(delta.device)
            right = self.right[name].to(delta.device)
            total = total + 0.5 * ((delta @ left) * (right @ delta)).sum()
        return loss + float(self.config.get("lambda", 1.0)) * total.to(loss.dtype), logits

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        if not hasattr(self, "left"):
            self._init_priors()
        device = next(self.model.parameters()).device
        layers = self._layers()
        names = [name for name, _ in layers]
        activations = collect(self.model, names, task.fit_loader, device, "input")
        grads = output_gradient_covariance(
            self.model,
            task.fit_loader,
            names,
            str(device),
            max_samples=int(self.config.get("kfac_samples", 256)),
            seed=int(self.config.get("kfac_seed", 0)),
        )
        for name, module in layers:
            update = {
                "left": input_second_moment(module, activations[name]),
                "right": grads[name].detach().double().cpu(),
            }
            if not (
                torch.isfinite(update["left"]).all() and torch.isfinite(update["right"]).all()
            ):
                continue
            if torch.trace(update["left"]) <= 0 or torch.trace(update["right"]) <= 0:
                continue
            merged = additive_nearest_kf(
                {"left": self.left[name], "right": self.right[name]}, update
            )
            self.left[name], self.right[name] = merged["left"], merged["right"]
