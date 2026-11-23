import numpy as np
import pandas as pd
import pytest

from crowd.aggregation import (
    DawidSkene,
    accuracy_to_log_odds,
    golden_accuracy,
    majority_vote,
    weighted_majority_vote,
)
from crowd.simulation import CrowdConfig, generate_crowd_dataset, with_spammer_share
from crowd.statistics import gini_coefficient, krippendorff_alpha_nominal, pairwise_agreement


def make_labels(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["task", "worker", "label"])


def test_majority_vote_picks_most_frequent_label() -> None:
    labels = make_labels([(0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 0, 0), (1, 1, 0), (1, 2, 0)])
    assert majority_vote(labels, n_classes=3).tolist() == [2, 0]


def test_majority_vote_breaks_ties_between_tied_classes_only() -> None:
    labels = make_labels([(task, worker, worker % 2) for task in range(200) for worker in range(2)])
    predicted = majority_vote(labels, n_classes=3)
    assert set(predicted.unique()) == {0, 1}


def test_weighted_majority_vote_follows_heavy_worker() -> None:
    labels = make_labels([(0, 0, 1), (0, 1, 2), (0, 2, 2)])
    weights = pd.Series({0: 5.0, 1: 1.0, 2: 1.0})
    assert weighted_majority_vote(labels, n_classes=3, worker_weights=weights).tolist() == [1]


def test_weighted_majority_vote_requires_all_weights() -> None:
    labels = make_labels([(0, 0, 1), (0, 1, 2)])
    with pytest.raises(ValueError):
        weighted_majority_vote(labels, n_classes=3, worker_weights=pd.Series({0: 1.0}))


def test_log_odds_weight_is_zero_at_chance_and_grows_with_accuracy() -> None:
    weights = accuracy_to_log_odds(pd.Series([0.1, 1 / 3, 0.6, 0.9]), n_classes=3)
    assert weights.iloc[0] == pytest.approx(0.0)
    assert weights.iloc[1] == pytest.approx(0.0)
    assert weights.iloc[2] < weights.iloc[3]


def test_golden_accuracy_is_smoothed_to_prior() -> None:
    labels = make_labels([(0, 0, 1), (1, 0, 1), (0, 1, 0), (2, 2, 0)])
    golden = pd.Series({0: 1, 1: 1})
    accuracy = golden_accuracy(labels, golden, prior_strength=1.0)
    prior = 2 / 3
    assert accuracy[0] == pytest.approx((2 + prior) / 3)
    assert accuracy[1] == pytest.approx(prior / 2)
    assert accuracy[2] == pytest.approx(prior)


def test_krippendorff_alpha_extremes() -> None:
    perfect = make_labels([(task, worker, task % 3) for task in range(30) for worker in range(3)])
    assert krippendorff_alpha_nominal(perfect, n_classes=3) == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    random_rows = [(task, worker, int(rng.integers(3))) for task in range(3000) for worker in range(3)]
    assert abs(krippendorff_alpha_nominal(make_labels(random_rows), n_classes=3)) < 0.03


def test_pairwise_agreement() -> None:
    labels = make_labels([(0, 0, 1), (0, 1, 1), (0, 2, 2), (1, 0, 0)])
    agreement = pairwise_agreement(labels, n_classes=3)
    assert agreement.tolist() == pytest.approx([1 / 3])


def test_gini_coefficient() -> None:
    assert gini_coefficient(np.ones(10)) == pytest.approx(0.0)
    assert gini_coefficient(np.array([0, 0, 0, 10])) == pytest.approx(0.75)


def test_with_spammer_share_keeps_proportions() -> None:
    config = with_spammer_share(CrowdConfig(), 0.4)
    shares = {t.name: t.share for t in config.worker_types}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["random_spammer"] + shares["constant_spammer"] == pytest.approx(0.4)


def test_simulation_is_reproducible_and_consistent() -> None:
    config = CrowdConfig(n_tasks=300, n_workers=50, seed=1)
    first = generate_crowd_dataset(config)
    second = generate_crowd_dataset(config)
    pd.testing.assert_frame_equal(first.labels, second.labels)
    assert not first.labels.duplicated(["task", "worker"]).any()
    np.testing.assert_allclose(first.confusions.sum(axis=2), 1.0)


def test_dawid_skene_beats_majority_vote_with_many_spammers() -> None:
    config = with_spammer_share(CrowdConfig(n_tasks=1500, n_workers=80, base_overlap=5, seed=2), 0.5)
    dataset = generate_crowd_dataset(config)
    truth = dataset.tasks.set_index("task")["true_label"]

    mv_accuracy = (majority_vote(dataset.labels, 3) == truth).mean()
    result = DawidSkene(n_classes=3).fit(dataset.labels)
    ds_accuracy = (result.labels == truth).mean()

    assert ds_accuracy > mv_accuracy
    assert np.all(np.diff(result.log_likelihoods) > -1e-6)
