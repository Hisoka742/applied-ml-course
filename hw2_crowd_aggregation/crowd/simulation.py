"""Generator of synthetic crowdsourcing data with a known ground truth.

The generative process mimics a typical three-class sentiment labelling project
on a crowdsourcing platform (Toloka / MTurk style):

* workers belong to several behavioural types: reliable, careless and spammers
  (random clickers and "always press the same button" workers);
* worker activity is heavy-tailed: a small core of workers does most of the job;
* errors are not uniform: mistakes go mostly to the adjacent ordinal class,
  and every honest worker has a personal bias towards some classes;
* tasks have different difficulty, and hard tasks cause *correlated* errors
  of honest workers (everybody is confused in the same way);
* overlap is dynamic (incremental relabelling): a task receives extra labels
  while the workers disagree;
* a small share of tasks are golden (control) tasks with a known answer that
  are shown to many workers, as on real platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

CLASS_NAMES: tuple[str, ...] = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class WorkerType:
    """Behavioural profile of a group of workers."""

    name: str
    share: float
    # Beta distribution of the worker accuracy; ignored for spammers.
    accuracy_alpha: float = 1.0
    accuracy_beta: float = 1.0
    # Spammers ignore the task content and do not slow down on hard tasks.
    is_spammer: bool = False
    # Constant spammers always choose one class, random ones click uniformly.
    is_constant: bool = False
    # Multiplier of the median answering time.
    speed_factor: float = 1.0


DEFAULT_WORKER_TYPES: tuple[WorkerType, ...] = (
    WorkerType("reliable", 0.70, accuracy_alpha=14.0, accuracy_beta=2.5),
    WorkerType("careless", 0.15, accuracy_alpha=5.0, accuracy_beta=4.0, speed_factor=0.7),
    WorkerType("random_spammer", 0.10, is_spammer=True, speed_factor=0.3),
    WorkerType("constant_spammer", 0.05, is_spammer=True, is_constant=True, speed_factor=0.25),
)


@dataclass(frozen=True)
class CrowdConfig:
    """Parameters of the simulated labelling project."""

    n_tasks: int = 4000
    n_workers: int = 300
    class_prior: tuple[float, ...] = (0.25, 0.30, 0.45)
    worker_types: tuple[WorkerType, ...] = DEFAULT_WORKER_TYPES
    golden_share: float = 0.05
    golden_overlap: int = 20
    base_overlap: int = 3
    max_overlap: int = 5
    # Extra labels are requested while the share of the top answer is below this value.
    agreement_threshold: float = 0.7
    # Sigma of the log-normal distribution of worker activity (heavier tail when larger).
    activity_sigma: float = 1.2
    difficulty_alpha: float = 1.2
    difficulty_beta: float = 4.0
    # Golden tasks are selected to be unambiguous, so their difficulty is lower.
    golden_difficulty_beta: float = 12.0
    # Decay rate of the error probability with the ordinal distance between classes.
    error_distance_decay: float = 1.0
    # Dirichlet concentration of personal class bias (smaller means stronger bias).
    bias_concentration: float = 5.0
    median_answer_time: float = 15.0
    seed: int = 0

    @property
    def n_classes(self) -> int:
        return len(self.class_prior)


@dataclass
class CrowdDataset:
    """Simulated labels together with the hidden ground truth."""

    # Columns: task, worker, label, answer_time.
    labels: pd.DataFrame
    # Columns: task, true_label, difficulty, is_golden.
    tasks: pd.DataFrame
    # Columns: worker, type, true_accuracy, activity.
    workers: pd.DataFrame
    # True confusion matrices of shape (n_workers, n_classes, n_classes), rows are true classes.
    confusions: np.ndarray
    config: CrowdConfig

    @property
    def golden_answers(self) -> pd.Series:
        """Known answers of control tasks, the only ground truth available in practice."""
        golden = self.tasks[self.tasks["is_golden"]]
        return golden.set_index("task")["true_label"]


def with_spammer_share(config: CrowdConfig, spammer_share: float) -> CrowdConfig:
    """Rescale worker type shares so that spammers make up ``spammer_share`` of the crowd."""
    honest_total = sum(t.share for t in config.worker_types if not t.is_spammer)
    spammer_total = sum(t.share for t in config.worker_types if t.is_spammer)
    worker_types = tuple(
        replace(t, share=t.share / spammer_total * spammer_share)
        if t.is_spammer
        else replace(t, share=t.share / honest_total * (1.0 - spammer_share))
        for t in config.worker_types
    )
    return replace(config, worker_types=worker_types)


def generate_crowd_dataset(config: CrowdConfig) -> CrowdDataset:
    """Simulate a labelling project described by ``config``."""
    rng = np.random.default_rng(config.seed)
    tasks = _sample_tasks(config, rng)
    workers, confusions = _sample_workers(config, rng)
    simulator = _AnswerSimulator(config, tasks, workers, confusions, rng)
    labels = simulator.collect_labels()
    return CrowdDataset(labels=labels, tasks=tasks, workers=workers, confusions=confusions, config=config)


def _sample_tasks(config: CrowdConfig, rng: np.random.Generator) -> pd.DataFrame:
    true_labels = rng.choice(config.n_classes, size=config.n_tasks, p=np.asarray(config.class_prior))
    difficulty = rng.beta(config.difficulty_alpha, config.difficulty_beta, size=config.n_tasks)

    n_golden = round(config.n_tasks * config.golden_share)
    is_golden = np.zeros(config.n_tasks, dtype=bool)
    is_golden[rng.choice(config.n_tasks, size=n_golden, replace=False)] = True
    difficulty[is_golden] = rng.beta(1.0, config.golden_difficulty_beta, size=n_golden)

    return pd.DataFrame(
        {
            "task": np.arange(config.n_tasks),
            "true_label": true_labels,
            "difficulty": difficulty,
            "is_golden": is_golden,
        }
    )


def _sample_workers(config: CrowdConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    n_classes = config.n_classes
    shares = np.array([t.share for t in config.worker_types])
    type_idx = rng.choice(len(config.worker_types), size=config.n_workers, p=shares / shares.sum())

    confusions = np.zeros((config.n_workers, n_classes, n_classes))
    for worker, idx in enumerate(type_idx):
        confusions[worker] = _sample_confusion(config, config.worker_types[idx], rng)

    prior = np.asarray(config.class_prior)
    true_accuracy = np.einsum("k,wkk->w", prior, confusions)
    workers = pd.DataFrame(
        {
            "worker": np.arange(config.n_workers),
            "type": [config.worker_types[idx].name for idx in type_idx],
            "true_accuracy": true_accuracy,
            "activity": rng.lognormal(0.0, config.activity_sigma, size=config.n_workers),
        }
    )
    return workers, confusions


def _sample_confusion(config: CrowdConfig, worker_type: WorkerType, rng: np.random.Generator) -> np.ndarray:
    n_classes = config.n_classes
    if worker_type.is_spammer:
        if worker_type.is_constant:
            confusion = np.zeros((n_classes, n_classes))
            confusion[:, rng.integers(n_classes)] = 1.0
            return confusion
        return np.full((n_classes, n_classes), 1.0 / n_classes)

    # An honest worker is at least as good as random guessing.
    accuracy = max(rng.beta(worker_type.accuracy_alpha, worker_type.accuracy_beta), 1.0 / n_classes)
    bias = rng.dirichlet(np.full(n_classes, config.bias_concentration))
    classes = np.arange(n_classes)

    confusion = np.zeros((n_classes, n_classes))
    for true_class in classes:
        # Errors prefer close ordinal classes and the classes the worker is biased to.
        error_weights = np.exp(-config.error_distance_decay * np.abs(classes - true_class)) * bias
        error_weights[true_class] = 0.0
        confusion[true_class] = (1.0 - accuracy) * error_weights / error_weights.sum()
        confusion[true_class, true_class] = accuracy
    return confusion


class _AnswerSimulator:
    """Assigns tasks to workers and draws their answers."""

    def __init__(
        self,
        config: CrowdConfig,
        tasks: pd.DataFrame,
        workers: pd.DataFrame,
        confusions: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self._config = config
        self._rng = rng
        self._confusions = confusions
        self._true_labels = tasks["true_label"].to_numpy()
        self._difficulty = tasks["difficulty"].to_numpy()
        self._is_golden = tasks["is_golden"].to_numpy()
        self._activity = workers["activity"].to_numpy() / workers["activity"].sum()

        type_by_name = {t.name: t for t in config.worker_types}
        worker_types = [type_by_name[name] for name in workers["type"]]
        self._is_spammer = np.array([t.is_spammer for t in worker_types])
        self._speed = np.array([t.speed_factor for t in worker_types])
        self._ambiguity = self._sample_ambiguity()

    def collect_labels(self) -> pd.DataFrame:
        records: list[pd.DataFrame] = []
        for task in range(self._config.n_tasks):
            if self._is_golden[task]:
                overlap = self._config.golden_overlap
            else:
                overlap = self._config.base_overlap
            assigned = self._pick_workers(overlap, exclude=np.array([], dtype=int))
            answers = self._answer(task, assigned)

            # Incremental relabelling: ask more workers while there is no clear majority.
            while (
                not self._is_golden[task]
                and len(assigned) < self._config.max_overlap
                and np.bincount(answers).max() / len(answers) < self._config.agreement_threshold
            ):
                extra = self._pick_workers(1, exclude=assigned)
                assigned = np.concatenate([assigned, extra])
                answers = np.concatenate([answers, self._answer(task, extra)])

            records.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "worker": assigned,
                        "label": answers,
                        "answer_time": self._answer_time(task, assigned),
                    }
                )
            )
        return pd.concat(records, ignore_index=True)

    def _sample_ambiguity(self) -> np.ndarray:
        """For every task, the answer distribution of an honest worker confused by it.

        A hard task is ambiguous between its true class and one adjacent class,
        so honest workers make the same mistake and their errors are correlated.
        """
        n_classes = self._config.n_classes
        shift = self._rng.choice([-1, 1], size=len(self._true_labels))
        confusing = self._true_labels + shift
        # Reflect the shift at the borders of the ordinal scale.
        out_of_range = (confusing < 0) | (confusing >= n_classes)
        confusing[out_of_range] = self._true_labels[out_of_range] - shift[out_of_range]

        ambiguity = np.zeros((len(self._true_labels), n_classes))
        rows = np.arange(len(self._true_labels))
        ambiguity[rows, self._true_labels] += 0.5
        ambiguity[rows, confusing] += 0.5
        return ambiguity

    def _pick_workers(self, n_workers: int, exclude: np.ndarray) -> np.ndarray:
        probabilities = self._activity.copy()
        probabilities[exclude] = 0.0
        return self._rng.choice(
            len(probabilities), size=n_workers, replace=False, p=probabilities / probabilities.sum()
        )

    def _answer(self, task: int, workers: np.ndarray) -> np.ndarray:
        probabilities = self._confusions[workers, self._true_labels[task]]
        # Only honest workers read the task, so only they are affected by its difficulty.
        confusion_weight = self._difficulty[task] * ~self._is_spammer[workers]
        probabilities = (1.0 - confusion_weight)[:, None] * probabilities + confusion_weight[
            :, None
        ] * self._ambiguity[task]
        # Inverse transform sampling of one class per row.
        uniform = self._rng.random((len(workers), 1))
        answers = (probabilities.cumsum(axis=1) < uniform).sum(axis=1)
        return np.minimum(answers, self._config.n_classes - 1)

    def _answer_time(self, task: int, workers: np.ndarray) -> np.ndarray:
        slowdown = 1.0 + self._difficulty[task] * ~self._is_spammer[workers]
        noise = self._rng.lognormal(0.0, 0.5, size=len(workers))
        return self._config.median_answer_time * self._speed[workers] * slowdown * noise
