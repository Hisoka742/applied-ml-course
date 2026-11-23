"""EuroSAT loading, semi-supervised splits and datasets.

Images are kept as a single uint8 tensor of the native 64x64 resolution:
cheap augmentations run on CPU at this size, while resizing to the model
resolution and normalization run on the GPU for the whole batch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as transforms_functional

CACHE_FILE_NAME = "eurosat_rgb_uint8.pt"


@dataclass
class ImageData:
    """All images of a dataset with their labels."""

    # uint8 tensor of shape (N, 3, H, W).
    images: torch.Tensor
    # int64 tensor of shape (N,).
    labels: torch.Tensor
    class_names: list[str]


@dataclass(frozen=True)
class SplitConfig:
    test_share: float = 0.2
    val_share: float = 0.05
    # Share of the remaining training pool that keeps its labels; the rest becomes unlabelled.
    labeled_share: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class DataSplits:
    """Indices of the four disjoint parts of the dataset."""

    labeled: np.ndarray
    # True labels of these images are hidden from training and used only to measure pseudo-label quality.
    unlabeled: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_eurosat(root: Path) -> ImageData:
    """Download EuroSAT RGB (27 000 images, 10 classes) and cache it as one tensor file."""
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / CACHE_FILE_NAME
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        return ImageData(cached["images"], cached["labels"], list(cached["class_names"]))

    dataset = torchvision.datasets.EuroSAT(str(root), download=True)
    images = torch.stack([transforms_functional.pil_to_tensor(image.convert("RGB")) for image, _ in dataset])
    labels = torch.tensor(dataset.targets, dtype=torch.int64)
    torch.save({"images": images, "labels": labels, "class_names": dataset.classes}, cache_path)
    return ImageData(images, labels, list(dataset.classes))


def make_splits(labels: torch.Tensor, config: SplitConfig) -> DataSplits:
    """Stratified split into labelled, unlabelled, validation and test parts."""
    indices = np.arange(len(labels))
    targets = labels.numpy()

    train_pool, test = train_test_split(
        indices, test_size=config.test_share, stratify=targets, random_state=config.seed
    )
    val_size = config.val_share / (1.0 - config.test_share)
    train_pool, val = train_test_split(
        train_pool, test_size=val_size, stratify=targets[train_pool], random_state=config.seed
    )
    labeled, unlabeled = train_test_split(
        train_pool, train_size=config.labeled_share, stratify=targets[train_pool], random_state=config.seed
    )
    return DataSplits(labeled=labeled, unlabeled=unlabeled, val=val, test=test)


class ImageTensorDataset(Dataset):
    """Images from a uint8 tensor with an optional per-sample transform."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels.")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class BatchPreprocessor(nn.Module):
    """Converts a uint8 batch to the model input: float, resized and normalized."""

    def __init__(self, image_size: int, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float() / 255.0
        if images.shape[-1] != self.image_size:
            images = functional.interpolate(
                images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        return (images - self.mean) / self.std
