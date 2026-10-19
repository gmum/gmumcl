"""InTAct baseline: interval-based activation consolidation.

Port of the official plugin `pkrukowski1/InTAct`
(commit 83b90c55f826f6ecbd1b635c9ab89e6ce291aec6),
`src/model/layer/interval_activation.py` and
`src/method/interval_penalization_big_model.py`:

- every ``nn.ReLU`` is replaced by ``IntervalActivation``: a LeakyReLU that
  records the current training batch's post-activation values; per-unit
  intervals are the 5th/95th percentiles of the task's activations, merged
  across tasks by element-wise min/max,
- ``var_scale``: per-batch activation variance inside each activation,
- ``lambda_int_drift``: output-preservation term computed on the next linear
  layer from the snapped parameters and the interval bounds (sign-split weight
  deltas times the bounds),
- ``lambda_feat``: drift of the first activation for units whose new value
  lies inside the old interval, against a frozen copy of the previous model,
- ``align_loss``: centre distance between non-overlapping old and new
  intervals, normalised by the mean radius.

Documented deviations: intervals are recomputed at each task end from the
task's loader (the official code replays inputs stored during the task at the
next task start; the model and data are the same), and units stay in their
flattened per-location form rather than being averaged over space.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..collect import collect
from ..plugin import MethodPluginABC


class IntervalActivation(nn.Module):
    """LeakyReLU that tracks per-unit activation intervals (official port)."""

    def __init__(self, lower_percentile: float = 0.05, upper_percentile: float = 0.95) -> None:
        super().__init__()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.min: Optional[torch.Tensor] = None
        self.max: Optional[torch.Tensor] = None
        self.curr_task_last_batch: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset_range(self, activations: List[torch.Tensor]) -> None:
        """Merge percentile bounds of ``activations`` into the stored interval."""
        if not activations:
            return
        stacked = torch.cat(activations, dim=0)
        sorted_values, _ = torch.sort(stacked, dim=0)
        samples = sorted_values.size(0)
        if samples == 0:
            return
        lower = max(0, min(int(samples * self.lower_percentile), samples - 1))
        upper = max(0, min(int(samples * self.upper_percentile), samples - 1))
        low = sorted_values[lower]
        high = sorted_values[upper]
        if self.min is None or self.max is None:
            self.min = low.clone()
            self.max = high.clone()
        else:
            self.min = torch.minimum(self.min, low)
            self.max = torch.maximum(self.max, high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x)
        if self.training:
            self.curr_task_last_batch = x
        return x


def _replace_relus(module: nn.Module) -> int:
    """Swap every ``nn.ReLU`` in ``module`` for an ``IntervalActivation``."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, IntervalActivation())
            replaced += 1
        else:
            replaced += _replace_relus(child)
    return replaced


class InTAct(MethodPluginABC):
    """Interval regularisation (see module docstring for the exact terms)."""

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.var_scale = float(self.config.get("var_scale", 0.01))
        self.lambda_int_drift = float(self.config.get("lambda_int_drift", 1.0))
        self.lambda_feat = float(self.config.get("lambda_feat", 1.0))
        self.use_repr_align_loss = bool(self.config.get("use_repr_align_loss", True))
        self.intervals: Dict[str, torch.Tensor] = {}
        self._swapped = False

    def set_model(self, model: nn.Module) -> None:
        super().set_model(model)
        if not self._swapped:
            _replace_relus(model)
            self._swapped = True

    def _activations(self) -> List[Tuple[str, IntervalActivation]]:
        assert self.model is not None
        return [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, IntervalActivation)
        ]

    def _next_linear(self, name: str) -> Optional[Tuple[str, nn.Linear]]:
        assert self.model is not None
        ordered = list(self.model.named_modules())
        position = next(index for index, (key, _) in enumerate(ordered) if key == name)
        for later_name, later in ordered[position + 1 :]:
            if isinstance(later, nn.Linear):
                return later_name, later
        return None

    def setup_task(self, task_id: int, task: Optional[Any] = None) -> None:
        assert self.model is not None
        self.task_id = task_id
        for _, module in self._activations():
            module.curr_task_last_batch = None
        self.old_module = None
        self.params_buffer: Dict[str, torch.Tensor] = {}
        if task_id > 0:
            self.old_module = copy.deepcopy(self.model).eval()
            for parameter in self.old_module.parameters():
                parameter.requires_grad = False
            self.params_buffer = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
            }

    def _old_activation(self, name: str, x: torch.Tensor) -> torch.Tensor:
        assert self.old_module is not None
        captured: Dict[str, torch.Tensor] = {}
        module = dict(self.old_module.named_modules())[name]

        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        handle = module.register_forward_hook(hook)
        with torch.no_grad():
            self.old_module(x)
        handle.remove()
        return captured[name]

    def forward(self, x, y, loss, logits):
        activations = self._activations()
        if not activations:
            return loss, logits
        var_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for _, module in activations:
            acts = module.curr_task_last_batch
            if acts is None:
                continue
            flat = acts.reshape(acts.shape[0], -1)
            var_loss = var_loss + flat.var(dim=0, unbiased=False).mean()

        interval_drift = torch.zeros((), dtype=loss.dtype, device=loss.device)
        output_reg = torch.zeros((), dtype=loss.dtype, device=loss.device)
        align_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)

        if self.task_id > 0 and self.intervals:
            first_name, first = activations[0]
            if first_name in self.intervals:
                acts = first.curr_task_last_batch
                old_acts = self._old_activation(first_name, x)
                low, high = self.intervals[first_name]
                low = low.to(acts.device)
                high = high.to(acts.device)
                mask = ((acts >= low) & (acts <= high)).float()
                interval_drift = (
                    (mask * (old_acts - acts).pow(2)).sum() / (mask.sum() + 1e-8)
                )

            for name, module in activations:
                if name not in self.intervals:
                    continue
                low, high = self.intervals[name]
                low = low.to(loss.device)
                high = high.to(loss.device)
                target = self._next_linear(name)
                if target is not None:
                    target_name, target_module = target
                    total_lower = torch.zeros(target_module.weight.shape[0], device=loss.device)
                    total_upper = torch.zeros(target_module.weight.shape[0], device=loss.device)
                    for parameter_name, parameter in target_module.named_parameters():
                        full_name = target_name + "." + parameter_name
                        if full_name not in self.params_buffer:
                            continue
                        previous = self.params_buffer[full_name].to(parameter.device)
                        if parameter_name == "weight":
                            delta = parameter - previous
                            positive = torch.relu(delta)
                            negative = torch.relu(-delta)
                            total_lower = total_lower + positive @ low - negative @ high
                            total_upper = total_upper + positive @ high - negative @ low
                        else:
                            total_lower = total_lower + (parameter - previous)
                            total_upper = total_upper + (parameter - previous)
                    output_reg = output_reg + total_lower.pow(2).mean() + total_upper.pow(2).mean()

                if self.use_repr_align_loss:
                    acts = module.curr_task_last_batch
                    if acts is None:
                        continue
                    flat = acts.reshape(acts.shape[0], -1)
                    if flat.shape[1] != low.shape[0]:
                        continue
                    previous_center = (high + low) / 2.0
                    previous_radii = (high - low) / 2.0
                    new_low = flat.min(dim=0).values
                    new_high = flat.max(dim=0).values
                    non_overlap = (new_low > high) | (new_high < low)
                    if bool(non_overlap.any()):
                        new_center = (new_high + new_low) / 2.0
                        align_loss = align_loss + torch.norm(
                            new_center[non_overlap] - previous_center[non_overlap], p=2
                        ) / (previous_radii.mean() + 1e-8)

        total = (
            loss
            + self.var_scale * var_loss
            + self.lambda_int_drift * output_reg
            + self.lambda_feat * interval_drift
            + align_loss
        )
        return total, logits

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        device = next(self.model.parameters()).device
        layers = self._activations()
        names = [name for name, _ in layers]
        if not names:
            return
        outputs = collect(self.model, names, task.fit_loader, device, capture="output")
        modules = dict(self.model.named_modules())
        for name, values in outputs.items():
            flat = values.reshape(values.shape[0], -1)
            module = modules[name]
            module.reset_range([flat])
            self.intervals[name] = torch.stack([module.min.clone(), module.max.clone()])
