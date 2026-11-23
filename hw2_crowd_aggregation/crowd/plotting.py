"""Plots shared by all experiments, so that every method is shown the same way."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrices(
    predictions: dict[str, pd.Series], true_labels: pd.Series, class_names: Sequence[str]
) -> Figure:
    """Row-normalized confusion matrix of every method in one row of subplots."""
    figure, axes = plt.subplots(1, len(predictions), figsize=(3.6 * len(predictions), 3.6))
    for ax, (method, predicted) in zip(axes, predictions.items(), strict=True):
        ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predicted.reindex(true_labels.index),
            labels=list(range(len(class_names))),
            display_labels=class_names,
            normalize="true",
            values_format=".2f",
            colorbar=False,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(method)
        ax.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    return figure


def plot_sweep(summary: pd.DataFrame, xlabel: str, title: str, ax: Axes) -> None:
    """Mean accuracy with a +-1 std band for every method (output of ``summarize_sweep``)."""
    x = summary.index.to_numpy()
    for method in summary["mean"].columns:
        mean = summary["mean"][method].to_numpy()
        std = summary["std"][method].to_numpy()
        ax.plot(x, mean, marker="o", label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
