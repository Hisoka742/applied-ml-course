"""Building datasets for the experiments and measuring the results consistently."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from eurosat_ssl.augmentations import build_augmentation
from eurosat_ssl.data import BatchPreprocessor, ImageData, ImageTensorDataset
from eurosat_ssl.models import ArchitectureSpec, create_model
from eurosat_ssl.training import TrainConfig, Trainer, TrainSetFactory, make_loader


@dataclass
class ExperimentResult:
    architecture: str
    setup: str
    # Per-epoch log of the whole run, see ``Trainer.history``.
    history: pd.DataFrame
    test_predictions: np.ndarray
    metrics: dict[str, float]


def train_set_factory(data: ImageData, indices: np.ndarray) -> TrainSetFactory:
    """Training set of the given images whose augmentation strength is chosen per stage."""
    images, labels = data.images[indices], data.labels[indices]

    def make_train_set(level: int) -> ImageTensorDataset:
        return ImageTensorDataset(images, labels, build_augmentation(level))

    return make_train_set


def eval_loader(data: ImageData, indices: np.ndarray, config: TrainConfig) -> DataLoader:
    dataset = ImageTensorDataset(data.images[indices], data.labels[indices])
    return make_loader(dataset, config, shuffle=False)


def class_prior(labels: torch.Tensor, n_classes: int) -> np.ndarray:
    return np.bincount(labels.numpy(), minlength=n_classes) / len(labels)


def make_trainer(
    spec: ArchitectureSpec,
    n_classes: int,
    image_size: int,
    config: TrainConfig,
    device: torch.device,
    pretrained: bool = True,
) -> Trainer:
    """Fresh pretrained model wrapped into a trainer with the matching input normalization."""
    torch.manual_seed(config.seed)
    model = create_model(spec, n_classes, image_size, pretrained=pretrained)
    normalization = model.pretrained_cfg
    preprocessor = BatchPreprocessor(image_size, normalization["mean"], normalization["std"])
    return Trainer(model, preprocessor, config, device)


def evaluate_on_test(
    trainer: Trainer, test_loader: DataLoader, test_labels: torch.Tensor, architecture: str, setup: str
) -> ExperimentResult:
    predictions = trainer.predict(test_loader)
    true_labels = test_labels.numpy()
    metrics = {
        "test_accuracy": accuracy_score(true_labels, predictions),
        "test_macro_f1": f1_score(true_labels, predictions, average="macro"),
        "epochs": len(trainer.history),
    }
    return ExperimentResult(architecture, setup, trainer.history_frame(), predictions, metrics)


def results_table(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = [{"architecture": r.architecture, "setup": r.setup, **r.metrics} for r in results]
    return pd.DataFrame(rows).set_index(["architecture", "setup"])
