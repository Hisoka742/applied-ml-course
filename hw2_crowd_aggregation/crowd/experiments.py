"""Running all aggregation methods on a dataset and measuring their quality."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from crowd.aggregation import (
    DawidSkene,
    accuracy_to_log_odds,
    agreement_accuracy,
    golden_accuracy,
    majority_vote,
    weighted_majority_vote,
)
from crowd.simulation import CrowdConfig, CrowdDataset, generate_crowd_dataset

METHOD_NAMES: tuple[str, ...] = (
    "MV",
    "WMV (agreement)",
    "WMV (golden)",
    "Dawid-Skene",
    "Dawid-Skene (golden)",
)


def aggregate_all(dataset: CrowdDataset, seed: int = 0) -> dict[str, pd.Series]:
    """Aggregated labels of every method, keyed by method name (see ``METHOD_NAMES``).

    Golden answers are the only ground truth the methods may use, as in practice.
    """
    labels = dataset.labels
    n_classes = dataset.config.n_classes
    golden_answers = dataset.golden_answers

    agreement_weights = accuracy_to_log_odds(agreement_accuracy(labels, n_classes, seed=seed), n_classes)
    golden_weights = accuracy_to_log_odds(golden_accuracy(labels, golden_answers), n_classes)
    dawid_skene = DawidSkene(n_classes)
    return {
        "MV": majority_vote(labels, n_classes, seed=seed),
        "WMV (agreement)": weighted_majority_vote(labels, n_classes, agreement_weights, seed=seed),
        "WMV (golden)": weighted_majority_vote(labels, n_classes, golden_weights, seed=seed),
        "Dawid-Skene": dawid_skene.fit(labels).labels,
        "Dawid-Skene (golden)": dawid_skene.fit(labels, known_labels=golden_answers).labels,
    }


def evaluation_tasks(dataset: CrowdDataset) -> pd.Series:
    """True labels of the tasks used for evaluation.

    Golden tasks are excluded: some methods use their answers, so measuring
    quality on them would be a leak.
    """
    tasks = dataset.tasks[~dataset.tasks["is_golden"]]
    return tasks.set_index("task")["true_label"]


def quality_table(predictions: dict[str, pd.Series], true_labels: pd.Series) -> pd.DataFrame:
    """Accuracy and macro F1 of every method on the tasks from ``true_labels``."""
    rows = {}
    for method, predicted in predictions.items():
        predicted = predicted.reindex(true_labels.index)
        rows[method] = {
            "accuracy": accuracy_score(true_labels, predicted),
            "macro_f1": f1_score(true_labels, predicted, average="macro"),
        }
    return pd.DataFrame(rows).T


def accuracy_by_group(
    predictions: dict[str, pd.Series], true_labels: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """Accuracy of every method inside each group of tasks; ``groups`` is indexed by task."""
    groups = groups.reindex(true_labels.index)
    columns = {}
    for method, predicted in predictions.items():
        is_correct = predicted.reindex(true_labels.index) == true_labels
        columns[method] = is_correct.groupby(groups, observed=True).mean()
    table = pd.DataFrame(columns)
    table.insert(0, "n_tasks", groups.value_counts().reindex(table.index))
    return table


def run_sweep(
    base_config: CrowdConfig,
    parameter_values: Iterable[float],
    make_config: Callable[[CrowdConfig, float], CrowdConfig],
    seeds: Iterable[int],
) -> pd.DataFrame:
    """Accuracy of every method for every parameter value and seed (long format)."""
    rows = []
    for value in parameter_values:
        for seed in seeds:
            config = replace(make_config(base_config, value), seed=seed)
            dataset = generate_crowd_dataset(config)
            quality = quality_table(aggregate_all(dataset, seed=seed), evaluation_tasks(dataset))
            for method, accuracy in quality["accuracy"].items():
                rows.append({"value": value, "seed": seed, "method": method, "accuracy": accuracy})
    return pd.DataFrame(rows)


def summarize_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of accuracy over seeds, methods in columns."""
    summary = sweep.groupby(["value", "method"])["accuracy"].agg(["mean", "std"]).unstack("method")
    return summary.reindex(columns=list(METHOD_NAMES), level="method")


def worker_quality_estimates(dataset: CrowdDataset) -> pd.DataFrame:
    """True worker accuracy next to its estimates by different methods."""
    labels = dataset.labels
    n_classes = dataset.config.n_classes
    estimates = pd.DataFrame(
        {
            "golden": golden_accuracy(labels, dataset.golden_answers),
            "agreement": agreement_accuracy(labels, n_classes),
            "dawid_skene": DawidSkene(n_classes).fit(labels).worker_accuracy,
        }
    )
    workers = dataset.workers.set_index("worker")[["type", "true_accuracy"]]
    workers["n_labels"] = labels.groupby("worker").size()
    return workers.join(estimates, how="inner")


def rank_correlations(estimates: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Spearman correlation between true worker accuracy and every estimate."""
    return pd.Series(
        {column: estimates["true_accuracy"].corr(estimates[column], method="spearman") for column in columns}
    )


def mean_absolute_errors(estimates: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    return pd.Series(
        {column: float(np.abs(estimates["true_accuracy"] - estimates[column]).mean()) for column in columns}
    )
