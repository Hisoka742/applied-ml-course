"""Label aggregation: majority vote, weighted majority vote and Dawid-Skene.

All methods take a long-format DataFrame with columns ``task``, ``worker`` and
``label`` (integer class index in ``[0, n_classes)``) and return a Series of
aggregated labels indexed by task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass(frozen=True)
class _EncodedLabels:
    task_codes: np.ndarray
    worker_codes: np.ndarray
    answers: np.ndarray
    tasks: pd.Index
    workers: pd.Index


def _encode(labels: pd.DataFrame, n_classes: int) -> _EncodedLabels:
    answers = labels["label"].to_numpy()
    if answers.min() < 0 or answers.max() >= n_classes:
        raise ValueError(f"Labels must be integers in [0, {n_classes}).")
    task_codes, tasks = pd.factorize(labels["task"], sort=True)
    worker_codes, workers = pd.factorize(labels["worker"], sort=True)
    return _EncodedLabels(task_codes, worker_codes, answers, pd.Index(tasks), pd.Index(workers))


def vote_scores(
    labels: pd.DataFrame, n_classes: int, worker_weights: pd.Series | None = None
) -> pd.DataFrame:
    """Sum of (weighted) votes for every class, shape ``(n_tasks, n_classes)``."""
    encoded = _encode(labels, n_classes)
    if worker_weights is None:
        vote_weights = np.ones(len(labels))
    else:
        missing = encoded.workers.difference(worker_weights.index)
        if len(missing) > 0:
            raise ValueError(f"No weight for {len(missing)} workers, e.g. {list(missing[:5])}.")
        vote_weights = worker_weights.reindex(encoded.workers).to_numpy()[encoded.worker_codes]

    scores = np.zeros((len(encoded.tasks), n_classes))
    np.add.at(scores, (encoded.task_codes, encoded.answers), vote_weights)
    return pd.DataFrame(scores, index=encoded.tasks.rename("task"))


def argmax_random_ties(scores: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Row-wise argmax that breaks ties uniformly at random instead of picking the first class.

    Always choosing the first class would silently bias the result towards it.
    """
    rng = np.random.default_rng(seed)
    values = scores.to_numpy()
    is_max = np.isclose(values, values.max(axis=1, keepdims=True))
    # Random keys restricted to the maximal classes; the largest key wins.
    random_keys = np.where(is_max, rng.random(values.shape), -1.0)
    return pd.Series(random_keys.argmax(axis=1), index=scores.index, name="label")


def majority_vote(labels: pd.DataFrame, n_classes: int, seed: int = 0) -> pd.Series:
    """Every worker has the same vote; ties are broken at random."""
    return argmax_random_ties(vote_scores(labels, n_classes), seed=seed)


def weighted_majority_vote(
    labels: pd.DataFrame, n_classes: int, worker_weights: pd.Series, seed: int = 0
) -> pd.Series:
    """Majority vote where the vote of a worker counts with its weight."""
    return argmax_random_ties(vote_scores(labels, n_classes, worker_weights), seed=seed)


def accuracy_to_log_odds(accuracy: pd.Series, n_classes: int, max_accuracy: float = 0.99) -> pd.Series:
    """Weight ``log((K - 1) * p / (1 - p))`` of a worker with accuracy ``p``.

    It is the Bayes-optimal vote weight when the worker errs uniformly over the
    other ``K - 1`` classes. Workers at or below the chance level get zero
    weight: for ``K > 2`` a negative vote for one class carries no useful
    information about which of the remaining classes is correct.
    """
    clipped = accuracy.clip(lower=1.0 / n_classes, upper=max_accuracy)
    return np.log((n_classes - 1) * clipped / (1.0 - clipped))


def golden_accuracy(
    labels: pd.DataFrame, golden_answers: pd.Series, prior_strength: float = 5.0
) -> pd.Series:
    """Accuracy of every worker on control tasks, smoothed towards the crowd average.

    A worker with ``n`` control answers of which ``c`` are correct gets
    ``(c + m * p0) / (n + m)``, where ``p0`` is the average accuracy on control
    tasks and ``m`` is ``prior_strength``. Workers without control answers get ``p0``.
    """
    counts = correct_answer_counts(labels, golden_answers)
    prior_mean = counts["correct"].sum() / counts["total"].sum()
    counts = counts.reindex(labels["worker"].unique(), fill_value=0).sort_index()
    return _smoothed_accuracy(counts, prior_mean, prior_strength)


def correct_answer_counts(labels: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Number of correct and of all answers of every worker on tasks that have a reference label.

    ``reference`` is indexed by task; workers without such tasks are absent from the result.
    """
    on_reference = labels[labels["task"].isin(reference.index)]
    is_correct = on_reference["label"].to_numpy() == reference.reindex(on_reference["task"]).to_numpy()
    answers = pd.DataFrame({"worker": on_reference["worker"].to_numpy(), "correct": is_correct})
    return answers.groupby("worker")["correct"].agg(correct="sum", total="count")


def _smoothed_accuracy(counts: pd.DataFrame, prior_mean: float, prior_strength: float) -> pd.Series:
    smoothed = (counts["correct"] + prior_strength * prior_mean) / (counts["total"] + prior_strength)
    return smoothed.rename("accuracy")


def agreement_accuracy(
    labels: pd.DataFrame,
    n_classes: int,
    n_iterations: int = 10,
    prior_strength: float = 5.0,
    seed: int = 0,
) -> pd.Series:
    """Worker accuracy estimated as agreement with the aggregate, without any ground truth.

    Starts from the plain majority vote and alternates between estimating
    worker accuracy against the current aggregate and re-aggregating with
    log-odds weights. The agreement is computed against an aggregate the
    worker voted for, so it is optimistic for workers on low-overlap tasks.
    """
    aggregated = majority_vote(labels, n_classes, seed=seed)
    accuracy = pd.Series(dtype=float)
    for _ in range(n_iterations):
        counts = correct_answer_counts(labels, aggregated)
        prior_mean = counts["correct"].sum() / counts["total"].sum()
        accuracy = _smoothed_accuracy(counts, prior_mean, prior_strength)

        weights = accuracy_to_log_odds(accuracy, n_classes)
        new_aggregated = weighted_majority_vote(labels, n_classes, weights, seed=seed)
        if new_aggregated.equals(aggregated):
            break
        aggregated = new_aggregated
    return accuracy


@dataclass
class DawidSkeneResult:
    """Fitted Dawid-Skene model."""

    # Posterior class probabilities, shape (n_tasks, n_classes), indexed by task.
    posteriors: pd.DataFrame
    # Estimated class prior, shape (n_classes,).
    class_prior: np.ndarray
    # Estimated confusion matrices, shape (n_workers, n_classes, n_classes); rows are true classes.
    confusions: np.ndarray
    workers: pd.Index
    log_likelihoods: list[float]

    @property
    def labels(self) -> pd.Series:
        return argmax_random_ties(self.posteriors)

    @property
    def worker_accuracy(self) -> pd.Series:
        """Expected accuracy of every worker under the estimated class prior."""
        accuracy = np.einsum("k,wkk->w", self.class_prior, self.confusions)
        return pd.Series(accuracy, index=self.workers.rename("worker"), name="accuracy")


class DawidSkene:
    """Dawid-Skene model fitted with the EM algorithm.

    Every worker is described by a full confusion matrix ``P(answer | true class)``,
    and answers of different workers are independent given the true class.
    E-step computes the posterior of the true class of every task, M-step
    re-estimates the class prior and the confusion matrices from the posteriors.

    Known labels (e.g. golden tasks) can be passed to ``fit``: their posteriors
    are clamped to the known class, which anchors the confusion matrices.
    """

    def __init__(
        self, n_classes: int, max_iter: int = 100, tol: float = 1e-6, smoothing: float = 0.01
    ) -> None:
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        # Additive (Laplace) smoothing that keeps probabilities away from zero.
        self.smoothing = smoothing

    def fit(self, labels: pd.DataFrame, known_labels: pd.Series | None = None) -> DawidSkeneResult:
        encoded = _encode(labels, self.n_classes)
        clamp_mask, clamp_values = self._known_label_posteriors(encoded.tasks, known_labels)

        # Initialization: soft majority vote (normalized vote counts).
        posteriors = vote_scores(labels, self.n_classes).to_numpy()
        posteriors /= posteriors.sum(axis=1, keepdims=True)
        posteriors[clamp_mask] = clamp_values

        log_likelihoods: list[float] = []
        class_prior = np.full(self.n_classes, 1.0 / self.n_classes)
        confusions = np.zeros((len(encoded.workers), self.n_classes, self.n_classes))
        for _ in range(self.max_iter):
            class_prior, confusions = self._m_step(encoded, posteriors)
            posteriors, log_likelihood = self._e_step(encoded, class_prior, confusions)
            posteriors[clamp_mask] = clamp_values
            log_likelihoods.append(log_likelihood)
            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol * abs(
                log_likelihoods[-2]
            ):
                break

        return DawidSkeneResult(
            posteriors=pd.DataFrame(posteriors, index=encoded.tasks.rename("task")),
            class_prior=class_prior,
            confusions=confusions,
            workers=encoded.workers,
            log_likelihoods=log_likelihoods,
        )

    def _known_label_posteriors(
        self, tasks: pd.Index, known_labels: pd.Series | None
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros(len(tasks), dtype=bool)
        if known_labels is None:
            return mask, np.zeros((0, self.n_classes))
        known = known_labels.reindex(tasks)
        mask = known.notna().to_numpy()
        values = np.eye(self.n_classes)[known[mask].astype(int).to_numpy()]
        return mask, values

    def _m_step(self, encoded: _EncodedLabels, posteriors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        class_prior = posteriors.sum(axis=0) + self.smoothing
        class_prior /= class_prior.sum()

        # confusions[w, k, l] accumulates P(true class of the task = k) over answers l of worker w.
        confusions = np.full((len(encoded.workers), self.n_classes, self.n_classes), self.smoothing)
        for answer in range(self.n_classes):
            is_answer = encoded.answers == answer
            np.add.at(
                confusions[:, :, answer],
                encoded.worker_codes[is_answer],
                posteriors[encoded.task_codes[is_answer]],
            )
        confusions /= confusions.sum(axis=2, keepdims=True)
        return class_prior, confusions

    def _e_step(
        self, encoded: _EncodedLabels, class_prior: np.ndarray, confusions: np.ndarray
    ) -> tuple[np.ndarray, float]:
        # Log-space computation avoids underflow on tasks with many labels.
        log_joint = np.tile(np.log(class_prior), (len(encoded.tasks), 1))
        answer_log_likelihoods = np.log(confusions[encoded.worker_codes, :, encoded.answers])
        np.add.at(log_joint, encoded.task_codes, answer_log_likelihoods)

        log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
        posteriors = np.exp(log_joint - log_evidence)
        return posteriors, float(log_evidence.sum())
