"""Weight-penalty baselines.

Sources:
- Naive: plain fine-tuning, no penalty (the lower bound).
- L2: quadratic penalty towards the task-end snapshot.
- EWC: Kirkpatrick et al. (2017); the diagonal-Fisher implementation follows
  gmumcl (MIT) and InTAct-Flex (MIT).
- EWC-DR: port of `scarlet0703/EWC-DR`
  (commit adcaaeb990111e02b084ef8c1e263e7cd00f5614), `models/ewc_dr.py`:
  importance with reversed logits (`logits * -1`, the LR trick), averaged over
  the task loader and clamped at `omegamax`, merged across tasks with
  alpha = known/total, and the quadratic penalty halved and sliced to the old
  head rows.
- SI: Zenke et al. (2017), path-integral importances, following gmumcl.
- MAS: Aljundi et al. (2018), output-norm importances.
- LwF: Li and Hoiem (2017), knowledge distillation to a frozen copy.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..growth import aligned_rows, padded_rows
from ..plugin import MethodPluginABC
from .common import diagonal_fisher, fisher_quadratic, parameter_snapshot


class Naive(MethodPluginABC):
    """Plain fine-tuning: no penalty, no distillation (the naive baseline)."""

    def forward(self, x, y, loss, logits):
        return loss, logits


class L2(MethodPluginABC):
    """lambda * ||theta - theta_0||^2, the L2 baseline."""

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        self.snapshot = parameter_snapshot(self.model)

    def forward(self, x, y, loss, logits):
        if not hasattr(self, "snapshot"):
            return loss, logits
        penalty = sum(
            ((aligned_rows(p, self.snapshot[n]) - self.snapshot[n]) ** 2).sum()
            for n, p in self.model.named_parameters()
        )
        return loss + float(self.config.get("lambda", 1.0)) * penalty, logits


class EWC(MethodPluginABC):
    """Diagonal-Fisher penalty. Modes: 'ewc', 'ewc_dr' (official port), 'online'."""

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.mode = str(self.config.get("mode", "ewc"))
        self.gamma = float(self.config.get("gamma", 1.0))
        self.max_batches = self.config.get("fisher_batches", None)
        self.seen_classes = 0

    def _ewc_dr_importance(self, loader) -> Dict[str, torch.Tensor]:
        """Official EWC-DR getImportance: reversed logits, averaged, clamped."""
        assert self.model is not None
        omegamax = float(self.config.get("omegamax", 1e-4))
        omega = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.train()
        batches = 0
        for x, y in loader:
            device = next(self.model.parameters()).device
            logits = self.model(x.to(device))
            logits = logits * -1  # LR
            loss = F.cross_entropy(logits, y.to(device))
            self.model.zero_grad()
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    omega[name] += parameter.grad.detach().pow(2)
            batches += 1
        self.model.zero_grad()
        for name in omega:
            omega[name] = (omega[name] / max(batches, 1)).clamp(max=omegamax)
        return omega

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        device = str(next(self.model.parameters()).device)
        if self.mode == "ewc_dr":
            new = self._ewc_dr_importance(task.fit_loader)
            if getattr(self, "omega", None) is None:
                self.omega = new
            else:
                total = self.seen_classes + len(task.classes)
                alpha = self.seen_classes / max(total, 1)
                for name in new:
                    old_rows = self.omega[name].shape[0]
                    new[name][:old_rows] = (
                        alpha * self.omega[name] + (1.0 - alpha) * new[name][:old_rows]
                    )
                self.omega = new
        else:
            fisher = diagonal_fisher(
                self.model, task.fit_loader, device, max_batches=self.max_batches
            )
            if self.mode == "online" and getattr(self, "fisher", None) is not None:
                fisher = {
                    n: self.gamma * padded_rows(self.fisher[n], fisher[n]) + fisher[n]
                    for n in fisher
                }
            self.fisher = fisher
        self.seen_classes += len(task.classes)
        self.mean = parameter_snapshot(self.model)

    def forward(self, x, y, loss, logits):
        if self.mode == "ewc_dr":
            if not hasattr(self, "omega"):
                return loss, logits
            penalty = torch.zeros((), dtype=loss.dtype, device=loss.device)
            for name, parameter in self.model.named_parameters():
                if name not in self.omega:
                    continue
                old = self.mean[name]
                difference = aligned_rows(parameter, old) - old
                penalty = penalty + (self.omega[name] * difference.pow(2)).sum() / 2.0
            return loss + float(self.config.get("lambda", 1.0)) * penalty, logits
        if not hasattr(self, "fisher"):
            return loss, logits
        penalty = fisher_quadratic(self.model, self.mean, self.fisher)
        if self.mode != "online":
            penalty = penalty / 2.0
        return loss + float(self.config.get("lambda", 1.0)) * penalty, logits


class SI(MethodPluginABC):
    """Synaptic Intelligence (Zenke et al., 2017); port follows gmumcl (MIT)."""

    def setup_task(self, task_id: int, task: Optional[Any] = None) -> None:
        assert self.model is not None
        self.star = getattr(self, "star", parameter_snapshot(self.model))
        self.omega = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.big_omega = getattr(self, "big_omega", None)
        self.theta_prev = parameter_snapshot(self.model)
        self.task_start = parameter_snapshot(self.model)

    def on_after_backward(self) -> None:
        assert self.model is not None
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            delta = parameter.detach() - self.theta_prev[name]
            self.omega[name] -= parameter.grad.detach() * delta
            self.theta_prev[name] = parameter.detach().clone()

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        xi = float(self.config.get("xi", 1e-6))
        if self.big_omega is None:
            self.big_omega = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        for name, parameter in self.model.named_parameters():
            star = self.star[name]
            anchor = padded_rows(star, parameter)
            if anchor.shape != star.shape:  # rows added for the current task
                anchor[star.shape[0] :] = self.task_start[name][star.shape[0] :]
            contribution = self.omega[name] / ((parameter.detach() - anchor) ** 2 + xi)
            self.big_omega[name] = padded_rows(self.big_omega[name], parameter) + contribution
        self.star = parameter_snapshot(self.model)

    def forward(self, x, y, loss, logits):
        if self.big_omega is None:
            return loss, logits
        penalty = sum(
            (self.big_omega[n] * (aligned_rows(p, self.big_omega[n]) - self.star[n]) ** 2).sum()
            for n, p in self.model.named_parameters()
        )
        return loss + float(self.config.get("lambda", 1.0)) * penalty, logits


class MAS(MethodPluginABC):
    """Memory Aware Synapses (Aljundi et al., 2018)."""

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        device = next(self.model.parameters()).device
        self.importance = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        for x, _ in task.fit_loader:
            self.model.zero_grad()
            objective = self.model(x.to(device)).pow(2).sum()
            gradients = torch.autograd.grad(
                objective, [p for p in self.model.parameters() if p.requires_grad]
            )
            for (name, _), gradient in zip(
                [(n, p) for n, p in self.model.named_parameters() if p.requires_grad], gradients
            ):
                self.importance[name] += gradient.detach().abs()
        self.star = parameter_snapshot(self.model)
        self.model.zero_grad()

    def forward(self, x, y, loss, logits):
        if not hasattr(self, "importance"):
            return loss, logits
        penalty = sum(
            (self.importance[n] * (aligned_rows(p, self.star[n]) - self.star[n]) ** 2).sum()
            for n, p in self.model.named_parameters()
        )
        return loss + float(self.config.get("lambda", 1.0)) * penalty, logits


class LwF(MethodPluginABC):
    """Learning without Forgetting (Li and Hoiem, 2017).

    The official `GMvandeVen/continual-learning` code weights the distillation
    term by ``context - 1`` (i.e. the number of previous tasks); a fixed
    lambda alone is too weak once several tasks have been seen.
    """

    def setup_task(self, task_id: int, task: Optional[Any] = None) -> None:
        assert self.model is not None
        self.task_id = task_id
        self.teacher = copy.deepcopy(self.model).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    def on_task_end(self, task_id: int, task) -> None:
        self.old_classes = getattr(self, "old_classes", 0) + len(task.classes)

    def forward(self, x, y, loss, logits):
        if getattr(self, "task_id", 0) == 0:
            return loss, logits
        temperature = float(self.config.get("temperature", 2.0))
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        old_classes = getattr(self, "old_classes", 0)
        if 0 < old_classes < logits.shape[1]:
            logits = logits[:, :old_classes]
            teacher_logits = teacher_logits[:, :old_classes]
        distillation = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * temperature * temperature
        weight = float(self.config.get("lambda", 1.0)) * self.task_id
        return loss + weight * distillation, logits
