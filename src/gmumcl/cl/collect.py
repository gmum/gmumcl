"""Activation collection through ActivationScope (PyPI package), input or output.

Unfolding convolutions and per-image scoring stay in our code; this module
returns one tensor per layer, concatenated over a loader.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


def collect(
    model: nn.Module,
    layer_names: Iterable[str],
    loader,
    device: str,
    capture: str = "input",
) -> Dict[str, torch.Tensor]:
    """Collect ``capture`` ('input' or 'output') activations of named layers.

    Returns a dict name -> tensor of shape (n_samples, ...) on CPU.
    """
    try:
        from activationscope import ActivationScope, CaptureMode, StoragePolicy
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "ActivationScope is required for activation collection; install activationscope"
        ) from error

    names: List[str] = list(layer_names)
    tracker = ActivationScope(storage=StoragePolicy.CPU, capture_mode=CaptureMode.SNAPSHOT)
    model.eval()
    with torch.no_grad():
        with tracker.track(model, include=names, capture=capture):
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(x.to(device))

    suffix = "." + capture
    result: Dict[str, torch.Tensor] = {}
    for key, tensors in tracker.activations.items():
        name = key[: -len(suffix)] if key.endswith(suffix) else key
        result[name] = torch.cat([t.detach().float().cpu() for t in tensors], dim=0)
    return result
