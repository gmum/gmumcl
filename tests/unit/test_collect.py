"""Unit tests for cl.collect: the streaming capped collector."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gmumcl.cl.collect import collect_capped, unfold_patches


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x)).flatten(1)
        return self.fc(x)


def _loader(samples=20, batch_size=5) -> DataLoader:
    inputs = torch.randn(samples, 3, 8, 8)
    labels = torch.zeros(samples, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size)


def test_capped_collection_bounds_rows_per_layer():
    model = TinyNet()
    points = collect_capped(
        model, ["conv", "fc"], _loader(), "cpu", max_points=12, patches_per_image=4
    )
    assert points["conv"].shape == (12, 27)
    assert points["fc"].shape == (12, 256)


def test_capped_collection_matches_the_unfold_without_a_cap():
    model = TinyNet()
    loader = _loader(samples=5, batch_size=5)
    points = collect_capped(model, ["conv", "fc"], loader, "cpu", patches_per_image=64)
    inputs = next(iter(loader))[0]
    patches = unfold_patches(inputs, 3, 3, stride=1, padding=1)
    assert torch.allclose(points["conv"], patches.reshape(-1, 27), atol=1e-6)
    assert points["fc"].shape[0] == 5
