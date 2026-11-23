"""Iterative pseudo-labelling (self-training) on the unlabelled part of the dataset.

Every round the current model labels the unlabelled pool with test-time
augmentation, the most confident predictions become pseudo-labels, and the
model is fine-tuned on labelled + pseudo-labelled images. The confidence
threshold decreases from round to round, so the pseudo-labelled set grows
as the model improves. To avoid the model drifting towards easy classes,
the number of pseudo-labels per class is capped proportionally to the class
distribution of the labelled set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from eurosat_ssl.augmentations import build_augmentation
from eurosat_ssl.data import ImageTensorDataset
from eurosat_ssl.training import Stage, Trainer, TrainSetFactory, make_loader


@dataclass(frozen=True)
class PseudoLabelingConfig:
    thresholds: tuple[float, ...] = (0.95, 0.9, 0.8)
    # A class may get at most ``slack`` times its labelled-set share of the selected pseudo-labels.
    class_balance_slack: float = 1.5
    use_tta: bool = True


@dataclass(frozen=True)
class PseudoLabels:
    # Positions inside the unlabelled pool.
    indices: np.ndarray
    labels: np.ndarray
    confidences: np.ndarray


def select_pseudo_labels(
    probabilities: torch.Tensor, threshold: float, class_prior: np.ndarray, slack: float
) -> PseudoLabels:
    """Confident predictions, at most ``ceil(prior_k * n_confident * slack)`` per class.

    Inside a class the most confident predictions are taken first.
    """
    confidences, predicted = probabilities.max(dim=1)
    confidences, predicted = confidences.numpy(), predicted.numpy()
    is_confident = confidences >= threshold
    caps = np.ceil(class_prior * is_confident.sum() * slack).astype(int)

    selected = []
    for class_index, cap in enumerate(caps):
        candidates = np.flatnonzero(is_confident & (predicted == class_index))
        most_confident_first = candidates[np.argsort(-confidences[candidates], kind="stable")]
        selected.append(most_confident_first[:cap])
    indices = np.sort(np.concatenate(selected))
    return PseudoLabels(indices=indices, labels=predicted[indices], confidences=confidences[indices])


def self_train(
    trainer: Trainer,
    config: PseudoLabelingConfig,
    student_stage: Stage,
    make_labeled_set: TrainSetFactory,
    unlabeled_images: torch.Tensor,
    hidden_labels: torch.Tensor,
    class_prior: np.ndarray,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> pd.DataFrame:
    """Runs all pseudo-labelling rounds on ``trainer`` and returns per-round statistics.

    ``hidden_labels`` are the true labels of the unlabelled pool. They never
    reach training and are used only to report the quality of pseudo-labels;
    test accuracy is likewise reported only and never used for model selection.
    """
    # Labels of the unlabelled loader are placeholders: only predictions are used.
    placeholder_labels = torch.zeros(len(unlabeled_images), dtype=torch.int64)
    unlabeled_loader = make_loader(
        ImageTensorDataset(unlabeled_images, placeholder_labels), trainer.config, shuffle=False
    )

    rounds = []
    for round_index, threshold in enumerate(config.thresholds, start=1):
        probabilities = trainer.predict_proba(unlabeled_loader, use_tta=config.use_tta)
        pseudo = select_pseudo_labels(probabilities, threshold, class_prior, config.class_balance_slack)
        pseudo_images = unlabeled_images[pseudo.indices]
        pseudo_targets = torch.from_numpy(pseudo.labels).long()

        def make_train_set(
            level: int, images: torch.Tensor = pseudo_images, targets: torch.Tensor = pseudo_targets
        ) -> Dataset:
            pseudo_set = ImageTensorDataset(images, targets, build_augmentation(level))
            return ConcatDataset([make_labeled_set(level), pseudo_set])

        trainer.fit(
            [replace(student_stage, name=f"pseudo-labels, round {round_index}")], make_train_set, val_loader
        )
        val_loss, val_accuracy = trainer.evaluate(val_loader)
        _, test_accuracy = trainer.evaluate(test_loader)
        rounds.append(
            {
                "round": round_index,
                "threshold": threshold,
                "n_pseudo_labels": len(pseudo.indices),
                "share_of_unlabeled_pool": len(pseudo.indices) / len(unlabeled_images),
                "pseudo_label_accuracy": pseudo_label_accuracy(pseudo, hidden_labels),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
            }
        )
    return pd.DataFrame(rounds)


def pseudo_label_accuracy(pseudo: PseudoLabels, hidden_labels: torch.Tensor) -> float:
    """Share of correct pseudo-labels; NaN when nothing passed the confidence threshold."""
    if len(pseudo.indices) == 0:
        return float("nan")
    return float((pseudo.labels == hidden_labels.numpy()[pseudo.indices]).mean())
