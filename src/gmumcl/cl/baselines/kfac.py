"""K-FAC baseline: Kronecker-factored Fisher.

Factor estimation is a port of the official implementation `Xin-Jing/K-FAC`
(commit 09f4565c6774a135175c1462a40c70da6df5d955), `K_FAC.py`, cross-checked
numerically against it at machine precision on an MLP:

- `A_hom[i,i] = a_hat^T a_hat / n` with the input augmented by a ones column
  (lines 125-135),
- `G_inc[i,i] = g^T g / n` where `g = predictions - onehot(sample)` is the
  gradient of the negative log-likelihood at *sampled* softmax targets
  (lines 118-131); the port differentiates `log_softmax` at the sampled target,
  which is `e_t - p` and gives the same outer product,
- the penalty is the Fisher quadratic of the update from the task-end
  snapshot, `sum_l tr(G_l dW_l A_l dW_l^T)` with the bias row/column included.

Documented deviations from the upstream code:

- upstream line 128 omits the `/ sampleminibatch_size` normalisation of
  `A_hom` for every layer except the first although its comment states it; the
  port applies the stated estimator to all layers,
- upstream estimates A and G on one random 1/8 sub-minibatch; the port
  estimates A on the fitting set and G on the first `kfac_samples` images,
- for convolutions the factors are averaged over output locations (a
  documented approximation for map-valued blocks).

The two-weight variant uses proposal Prop. 15 on the update:
`lambda_c * alpha^2 + lambda_s * gamma^2` per unit with `alpha = u^T mu + db`
and `gamma = ||Sigma^{1/2} u||`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..collect import collect
from ..growth import aligned_rows
from ..plugin import MethodPluginABC
from .common import input_second_moment, output_gradient_covariance, parameter_snapshot


class KFAC(MethodPluginABC):
    """Fisher quadratic with Kronecker factors (see module docstring)."""

    def on_task_end(self, task_id: int, task) -> None:
        assert self.model is not None
        device = next(self.model.parameters()).device
        layer_names = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        activations = collect(self.model, layer_names, task.fit_loader, device, "input")
        modules = dict(self.model.named_modules())
        self.inputs = {
            name: input_second_moment(modules[name], activations[name]) for name in layer_names
        }
        self.output_grads = output_gradient_covariance(
            self.model,
            task.fit_loader,
            layer_names,
            str(device),
            max_samples=int(self.config.get("kfac_samples", 256)),
            seed=int(self.config.get("kfac_seed", 0)),
        )
        self.mean = parameter_snapshot(self.model)

    def _quadratic(self, name: str, module: nn.Module) -> torch.Tensor:
        mean_weight = self.mean[name + ".weight"]
        weight = (aligned_rows(module.weight, mean_weight) - mean_weight).reshape(
            mean_weight.shape[0], -1
        )
        if module.bias is not None:
            mean_bias = self.mean[name + ".bias"]
            bias = aligned_rows(module.bias, mean_bias) - mean_bias
        else:
            bias = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
        delta = torch.cat([weight, bias.unsqueeze(1)], dim=1)
        gram = self.output_grads.get(name)
        if gram is None:
            return torch.zeros((), dtype=torch.float64, device=weight.device)
        factor = self.inputs[name].to(weight.device)
        value = delta.double() @ factor @ delta.double().transpose(0, 1)
        return (gram * value).sum()

    def _two_weight(self, name: str, module: nn.Module) -> torch.Tensor:
        factor = self.inputs[name].to(module.weight.device)
        mean = factor[:-1, -1]
        covariance = factor[:-1, :-1] - torch.outer(mean, mean)
        mean_weight = self.mean[name + ".weight"]
        weight = (
            (aligned_rows(module.weight, mean_weight) - mean_weight)
            .reshape(mean_weight.shape[0], -1)
            .double()
        )
        if module.bias is not None:
            mean_bias = self.mean[name + ".bias"]
            bias = (aligned_rows(module.bias, mean_bias) - mean_bias).double()
        else:
            bias = torch.zeros(weight.shape[0], dtype=torch.float64, device=weight.device)
        center = weight @ mean + bias
        spread = (weight @ covariance * weight).sum(dim=1)
        lambda_center = float(self.config.get("lambda_center", 1.0))
        lambda_shape = float(self.config.get("lambda_shape", 1.0))
        return lambda_center * (center**2).sum() + lambda_shape * spread.sum()

    def forward(self, x, y, loss, logits):
        if not hasattr(self, "inputs"):
            return loss, logits
        total = torch.zeros((), dtype=torch.float64, device=loss.device)
        for name, module in self.model.named_modules():
            if name not in self.inputs:
                continue
            if bool(self.config.get("two_weight", False)):
                total = total + self._two_weight(name, module)
            else:
                total = total + self._quadratic(name, module)
        if bool(self.config.get("two_weight", False)):
            return loss + total.to(loss.dtype), logits
        return loss + float(self.config.get("lambda", 1.0)) * total.to(loss.dtype), logits
