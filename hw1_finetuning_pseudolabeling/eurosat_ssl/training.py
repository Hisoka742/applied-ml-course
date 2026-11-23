"""Staged fine-tuning: iterative unfreezing, adaptive learning rates and growing augmentation.

A training run is a sequence of stages. Each stage unfreezes more layer
groups (counting from the head), may switch to a stronger augmentation
level, and builds a fresh optimizer with discriminative learning rates:
the further a group is from the head, the smaller its learning rate. Inside a
stage the learning rate adapts to the validation loss (ReduceLROnPlateau),
and the stage stops early when the validation loss stops improving. Every
stage starts from the best weights found so far.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from eurosat_ssl.augmentations import tta_views
from eurosat_ssl.data import BatchPreprocessor
from eurosat_ssl.models import layer_groups

# Parameters that are embeddings rather than weights and must not be decayed.
_NO_DECAY_NAME_SUFFIXES = ("cls_token", "reg_token", "pos_embed")


@dataclass(frozen=True)
class Stage:
    name: str
    # Number of layer groups to train, counted from the head; None trains the whole network.
    n_unfrozen_groups: int | None
    augmentation_level: int
    base_lr: float
    max_epochs: int
    # Learning rate multiplier per layer group away from the head (1.0 means one lr for all layers).
    lr_decay: float = 0.5
    # Early stopping: the stage ends after this many epochs without validation loss improvement.
    patience: int = 3


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    num_workers: int = 2
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    plateau_factor: float = 0.3
    plateau_patience: int = 1
    min_improvement: float = 1e-4
    seed: int = 0


# Builds the training set for a given augmentation level.
TrainSetFactory = Callable[[int], Dataset]


def make_loader(dataset: Dataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        # Dropping the last incomplete batch keeps batch statistics of training steps stable.
        drop_last=shuffle and len(dataset) > config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


class Trainer:
    """Trains one model through any number of stages and remembers the best weights."""

    def __init__(
        self, model: nn.Module, preprocessor: BatchPreprocessor, config: TrainConfig, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.preprocessor = preprocessor.to(device)
        self.config = config
        self.device = device
        self.use_amp = device.type == "cuda"
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        self.groups = layer_groups(model)
        self.history: list[dict[str, float | int | str]] = []
        self.best_val_loss = math.inf
        self.best_state: dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())

    def fit(self, stages: Sequence[Stage], make_train_set: TrainSetFactory, val_loader: DataLoader) -> None:
        for stage in stages:
            train_loader = make_loader(make_train_set(stage.augmentation_level), self.config, shuffle=True)
            self._run_stage(stage, train_loader, val_loader)
            # The next stage (or the caller) continues from the best weights found so far.
            self.model.load_state_dict(self.best_state)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Mean loss and accuracy on a labelled loader."""
        self.model.eval()
        total_loss, n_correct, n_samples = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(self.preprocessor(images))
                loss = self.criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_samples += len(labels)
        return total_loss / n_samples, n_correct / n_samples

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader, use_tta: bool = False) -> torch.Tensor:
        """Class probabilities of shape (N, n_classes) on CPU; TTA averages over orientation views."""
        self.model.eval()
        batches = []
        for images, _ in loader:
            images = images.to(self.device)
            views = tta_views(images) if use_tta else [images]
            view_probabilities = []
            for view in views:
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(self.preprocessor(view))
                view_probabilities.append(logits.float().softmax(dim=1))
            batches.append(torch.stack(view_probabilities).mean(dim=0).cpu())
        return torch.cat(batches)

    def predict(self, loader: DataLoader, use_tta: bool = False) -> np.ndarray:
        return self.predict_proba(loader, use_tta).argmax(dim=1).numpy()

    def _run_stage(self, stage: Stage, train_loader: DataLoader, val_loader: DataLoader) -> None:
        n_unfrozen = self._set_trainable(stage.n_unfrozen_groups)
        optimizer = self._make_optimizer(stage, n_unfrozen)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.config.plateau_factor, patience=self.config.plateau_patience
        )

        stage_best_loss = math.inf
        epochs_without_improvement = 0
        for stage_epoch in range(1, stage.max_epochs + 1):
            # The head group always has the largest learning rate; it is logged as "lr".
            head_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy = self._train_epoch(train_loader, optimizer)
            val_loss, val_accuracy = self.evaluate(val_loader)
            scheduler.step(val_loss)

            self.history.append(
                {
                    "epoch": len(self.history) + 1,
                    "stage": stage.name,
                    "stage_epoch": stage_epoch,
                    "unfrozen_groups": n_unfrozen,
                    "augmentation_level": stage.augmentation_level,
                    "lr": head_lr,
                    "n_train": len(train_loader.dataset),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
            if val_loss < stage_best_loss - self.config.min_improvement:
                stage_best_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= stage.patience:
                    break

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        self.model.train()
        total_loss, n_correct, n_samples = 0.0, 0, 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(self.preprocessor(images))
                loss = self.criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(labels)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_samples += len(labels)
        return total_loss / n_samples, n_correct / n_samples

    def _set_trainable(self, n_unfrozen_groups: int | None) -> int:
        n_unfrozen = (
            len(self.groups) if n_unfrozen_groups is None else min(n_unfrozen_groups, len(self.groups))
        )
        for index, group in enumerate(self.groups):
            for _, parameter in group:
                parameter.requires_grad_(index < n_unfrozen)
        return n_unfrozen

    def _make_optimizer(self, stage: Stage, n_unfrozen: int) -> AdamW:
        param_groups = []
        for index, group in enumerate(self.groups[:n_unfrozen]):
            lr = stage.base_lr * stage.lr_decay**index
            # Norm layers, biases and embeddings are not decayed: weight decay hurts them.
            decayed = [p for name, p in group if p.ndim > 1 and not name.endswith(_NO_DECAY_NAME_SUFFIXES)]
            not_decayed = [p for name, p in group if p.ndim <= 1 or name.endswith(_NO_DECAY_NAME_SUFFIXES)]
            param_groups.append({"params": decayed, "lr": lr, "weight_decay": self.config.weight_decay})
            param_groups.append({"params": not_decayed, "lr": lr, "weight_decay": 0.0})
        return AdamW([group for group in param_groups if group["params"]])
