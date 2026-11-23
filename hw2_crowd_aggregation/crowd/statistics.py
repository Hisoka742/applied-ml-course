"""Descriptive statistics of crowdsourced labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def labels_per_task(labels: pd.DataFrame) -> pd.Series:
    return labels.groupby("task").size().rename("n_labels")


def labels_per_worker(labels: pd.DataFrame) -> pd.Series:
    return labels.groupby("worker").size().rename("n_labels")


def gini_coefficient(values: np.ndarray | pd.Series) -> float:
    """Gini coefficient of non-negative values: 0 is perfect equality, 1 is maximal inequality."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    n = len(sorted_values)
    ranks = np.arange(1, n + 1)
    return float((2 * ranks - n - 1) @ sorted_values / (n * sorted_values.sum()))


def lorenz_curve(values: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Share of workers (x) vs. share of labels they made (y), least active workers first."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    x = np.linspace(0.0, 1.0, len(sorted_values) + 1)
    y = np.concatenate([[0.0], sorted_values.cumsum() / sorted_values.sum()])
    return x, y


def top_workers_share(values: np.ndarray | pd.Series, top_fraction: float) -> float:
    """Share of all labels made by the ``top_fraction`` most active workers."""
    sorted_values = np.sort(np.asarray(values, dtype=float))[::-1]
    n_top = max(1, round(len(sorted_values) * top_fraction))
    return float(sorted_values[:n_top].sum() / sorted_values.sum())


def task_vote_counts(labels: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """Number of votes for every class on every task, shape ``(n_tasks, n_classes)``."""
    counts = pd.crosstab(labels["task"], labels["label"])
    return counts.reindex(columns=range(n_classes), fill_value=0)


def pairwise_agreement(labels: pd.DataFrame, n_classes: int) -> pd.Series:
    """Share of agreeing pairs of workers on every task with at least two labels."""
    counts = task_vote_counts(labels, n_classes)
    n_labels = counts.sum(axis=1)
    counts = counts[n_labels >= 2]
    n_labels = n_labels[n_labels >= 2]
    agreeing_pairs = (counts * (counts - 1)).sum(axis=1)
    return (agreeing_pairs / (n_labels * (n_labels - 1))).rename("pairwise_agreement")


def krippendorff_alpha_nominal(labels: pd.DataFrame, n_classes: int) -> float:
    """Krippendorff's alpha for nominal data; supports a varying number of labels per task.

    ``alpha = 1 - D_o / D_e``, where ``D_o`` is the observed disagreement
    within tasks and ``D_e`` is the disagreement expected by chance given
    the overall label distribution. 1 is perfect agreement, 0 is chance level.
    """
    counts = task_vote_counts(labels, n_classes).to_numpy(dtype=float)
    n_labels = counts.sum(axis=1)
    counts = counts[n_labels >= 2]
    n_labels = n_labels[n_labels >= 2]

    # Coincidence matrix: every ordered pair of labels within a task, weighted by 1 / (m_u - 1).
    pair_counts = counts[:, :, None] * counts[:, None, :]
    pair_counts[:, np.arange(n_classes), np.arange(n_classes)] -= counts
    coincidences = (pair_counts / (n_labels - 1)[:, None, None]).sum(axis=0)

    class_totals = coincidences.sum(axis=1)
    total = class_totals.sum()
    observed_disagreement = coincidences.sum() - np.trace(coincidences)
    expected_disagreement = (total**2 - (class_totals**2).sum()) / (total - 1)
    return float(1.0 - observed_disagreement / expected_disagreement)


def label_confusion(labels: pd.DataFrame, tasks: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """Row-normalized confusion between true labels and individual worker answers."""
    merged = labels.merge(tasks[["task", "true_label"]], on="task")
    confusion = pd.crosstab(merged["true_label"], merged["label"], normalize="index")
    return confusion.reindex(index=range(n_classes), columns=range(n_classes), fill_value=0.0)
