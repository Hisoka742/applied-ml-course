"""Plots shared by all experiments, so that every run is shown the same way."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay


def plot_run(
    history: pd.DataFrame,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Sequence[str],
    title: str,
) -> Figure:
    """Loss and accuracy curves with stage boundaries, plus the test confusion matrix."""
    figure, axes = plt.subplot_mosaic(
        [["loss", "accuracy"], ["confusion", "confusion"]], figsize=(13, 13), height_ratios=[1, 1.9]
    )
    for metric in ("loss", "accuracy"):
        ax = axes[metric]
        ax.plot(history["epoch"], history[f"train_{metric}"], marker=".", label="train")
        ax.plot(history["epoch"], history[f"val_{metric}"], marker=".", label="validation")
        _mark_stages(ax, history)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes["loss"].set_title("Loss (dashed lines: stage boundaries)")
    axes["accuracy"].set_title("Accuracy")

    display = ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        display_labels=class_names,
        normalize="true",
        values_format=".2f",
        colorbar=False,
        cmap="Blues",
        ax=axes["confusion"],
    )
    for text in display.text_.ravel():
        text.set_fontsize(9)
    axes["confusion"].set_title("Test confusion matrix (row-normalized)")
    axes["confusion"].tick_params(axis="x", labelrotation=45)

    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    return figure


def _mark_stages(ax: Axes, history: pd.DataFrame) -> None:
    # A new stage starts where the stage name changes.
    starts = history.index[history["stage"] != history["stage"].shift()]
    for start in starts[1:]:
        ax.axvline(history.loc[start, "epoch"] - 0.5, color="grey", linestyle="--", alpha=0.6)
