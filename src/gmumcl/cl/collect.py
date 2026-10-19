"""Activation collection through ActivationScope (PyPI package), input or output.

Unfolding convolutions and per-image scoring stay in our code; this module
returns one tensor per layer, concatenated over a loader. The capped variant
streams the loader so convolutional patch products stay bounded.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def unfold_patches(
    x: torch.Tensor, kernel_height: int, kernel_width: int, stride: int = 1, padding: int = 0
) -> torch.Tensor:
    """Unfold an input map (N, C, H, W) into patches (N, L, D)."""
    patches = F.unfold(x, (kernel_height, kernel_width), stride=stride, padding=padding)
    return patches.transpose(1, 2)


def sample_patches(
    patches: torch.Tensor, max_per_image: int = 16, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample at most ``max_per_image`` locations per image: (N, L, D) -> (n, D)."""
    images, locations, dimension = patches.shape
    count = min(locations, max_per_image)
    if count < locations:
        index = torch.randperm(locations, generator=generator, device=patches.device)[:count]
    else:
        index = torch.arange(locations, device=patches.device)
    return patches[:, index, :].reshape(images * count, dimension)


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


def collect_capped(
    model: nn.Module,
    layer_names: Iterable[str],
    loader,
    device: str,
    capture: str = "input",
    max_points: Optional[int] = None,
    patches_per_image: int = 16,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Streaming collection with at most ``max_points`` region rows per layer.

    Same ActivationScope capture as :func:`collect`, drained batch by batch:
    convolution inputs are unfolded per batch with at most
    ``patches_per_image`` locations per image, and rows are kept by random
    priority, i.e. a uniform sample of the stream. Bounds host memory where
    the plain collector would materialise every patch of every sample.
    """
    try:
        from activationscope import ActivationScope, CaptureMode, StoragePolicy
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "ActivationScope is required for activation collection; install activationscope"
        ) from error

    modules = dict(model.named_modules())
    names: List[str] = [name for name in layer_names if name in modules]
    generator = generator or torch.Generator()
    suffix = "." + capture
    tracker = ActivationScope(storage=StoragePolicy.CPU, capture_mode=CaptureMode.SNAPSHOT)
    buffers: Dict[str, torch.Tensor] = {}
    priorities: Dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        with tracker.track(model, include=names, capture=capture):
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(x.to(device))
                for key, tensors in tracker.activations.items():
                    name = key[: -len(suffix)] if key.endswith(suffix) else key
                    module = modules.get(name)
                    if module is None:
                        continue
                    chunk = torch.cat([t.detach().float().cpu() for t in tensors], dim=0)
                    if isinstance(module, nn.Conv2d) and capture == "input":
                        patches = unfold_patches(
                            chunk,
                            module.kernel_size[0],
                            module.kernel_size[1],
                            stride=module.stride[0],
                            padding=module.padding[0],
                        )
                        rows = sample_patches(patches, patches_per_image, generator)
                    else:
                        rows = chunk.reshape(chunk.shape[0], -1)
                    keys = torch.rand(rows.shape[0], generator=generator)
                    if name in buffers:
                        rows = torch.cat([buffers[name], rows])
                        keys = torch.cat([priorities[name], keys])
                    limit = rows.shape[0] if max_points is None else min(int(max_points), rows.shape[0])
                    if limit < rows.shape[0]:
                        index = torch.topk(keys, limit, largest=False).indices
                        rows = rows[index]
                        keys = keys[index]
                    buffers[name] = rows
                    priorities[name] = keys
                tracker.clear()
    return buffers
