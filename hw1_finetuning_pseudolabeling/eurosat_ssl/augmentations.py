"""Augmentation levels of increasing strength for iterative augmentation.

Satellite images have no canonical orientation, so flips and 90-degree
rotations never change the class and are safe from the first level on.
All transforms work on uint8 tensors of the native 64x64 resolution.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import v2

AUGMENTATION_LEVEL_NAMES: tuple[str, ...] = ("none", "geometric", "geometric + crop + color", "strong")
NATIVE_IMAGE_SIZE = 64


class RandomRot90(nn.Module):
    """Rotation by a random multiple of 90 degrees: lossless, unlike arbitrary-angle rotation."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        n_turns = int(torch.randint(0, 4, (1,)))
        return torch.rot90(image, k=n_turns, dims=(-2, -1))


def build_augmentation(level: int) -> v2.Compose:
    """Training transform of the given strength, from 0 (no augmentation) to 3 (strongest)."""
    if not 0 <= level < len(AUGMENTATION_LEVEL_NAMES):
        raise ValueError(f"Augmentation level must be in [0, {len(AUGMENTATION_LEVEL_NAMES)}), got {level}.")

    transforms: list[nn.Module] = []
    if level >= 1:
        transforms += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), RandomRot90()]
    if level >= 2:
        transforms += [
            v2.RandomResizedCrop(NATIVE_IMAGE_SIZE, scale=(0.6, 1.0), antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        ]
    if level >= 3:
        transforms += [
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
        ]
    if not transforms:
        return v2.Compose([v2.Identity()])
    return v2.Compose(transforms)


def tta_views(images: torch.Tensor) -> list[torch.Tensor]:
    """Orientation views of a batch for test-time augmentation (identity, flips, 90-degree rotation)."""
    return [
        images,
        torch.flip(images, dims=(-1,)),
        torch.flip(images, dims=(-2,)),
        torch.rot90(images, k=1, dims=(-2, -1)),
    ]
