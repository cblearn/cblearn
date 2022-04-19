from cblearn.datasets import make_random_triplets
from cblearn.cluster import ComparisonHC
from .._comparison_hc import triplets_to_quadruplets, unify_triplet_order, reduce_triplets
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_blobs
import numpy as np


def generate_gmm_data(n, means: np.ndarray, std: float, seed: int):
    stds = std * np.ones(means.shape)
    num_clusters = len(means)

    return make_blobs(n_samples=[n] * num_clusters, centers=means, cluster_std=stds, n_features=2,
                      random_state=seed, shuffle=False)


def test_chc_performance():
    """Checks if chc can cluster a very simple problem."""
    seed = 2
    xs, ys = generate_gmm_data(
        10, np.array([[1, 0], [-1, 0]]), 0.2, seed)
    t, r = make_random_triplets(xs, result_format="list-boolean", size=5000, random_state=seed)
    chc = ComparisonHC(2)

    # testing score method
    y_chc = chc.fit_predict(t, r)
    score = normalized_mutual_info_score(y_chc, ys)
    assert score > 0.99


def test_chc_score_equalities():
    """Checks if the different methods of obtaining a clustering in ComparisonHC
    give the same output."""
    seed = 2
    xs, ys = generate_gmm_data(
        10, np.array([[1, 0], [-1, 0]]), 0.2, seed)
    t, r = make_random_triplets(xs, result_format="list-boolean", size=5000, random_state=seed)
    chc = ComparisonHC(2)
    y_chc = chc.fit_predict(t, r)
    score = normalized_mutual_info_score(y_chc, ys)
    assert score == chc.score((t, r), ys)

    # testing score on unified triplets
    unified_triplets = unify_triplet_order(t, r)
    y_chc_unified = chc.fit_predict(unified_triplets)
    score_unified = normalized_mutual_info_score(y_chc_unified, ys)
    assert score_unified == score
    assert np.all(y_chc == y_chc_unified)
    assert score_unified == chc.score(unified_triplets, ys)


def test_quadruplet_generation():
    triplets = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    responses = np.array([1, 1, 0])
    quad_responses = triplets_to_quadruplets(triplets, responses)
    assert quad_responses[0, 1, 0, 2] == 1
    assert quad_responses[1, 2, 1, 3] == 1
    assert quad_responses[2, 3, 2, 4] == -1
    assert quad_responses[0, 2, 0, 1] == -1
    assert quad_responses[1, 3, 1, 2] == -1
    assert quad_responses[2, 4, 2, 3] == 1

    quad_no_responses = triplets_to_quadruplets(triplets)
    assert quad_no_responses[0, 1, 0, 2] == 1
    assert quad_no_responses[1, 2, 1, 3] == 1
    assert quad_no_responses[2, 3, 2, 4] == 1
    assert quad_no_responses[0, 2, 0, 1] == -1
    assert quad_no_responses[1, 3, 1, 2] == -1
    assert quad_no_responses[2, 4, 2, 3] == -1

    assert (quad_responses == 1).sum() == 3
    assert (quad_responses == -1).sum() == 3
    assert quad_responses.sum() == 0

    assert (quad_no_responses == 1).sum() == 3
    assert (quad_no_responses == -1).sum() == 3
    assert quad_no_responses.sum() == 0


def test_normal_triplets_unify():
    triplets = np.array(
        [[0, 1, 2], [1, 2, 3], [0, 2, 1], [0, 1, 2], [3, 2, 1]])
    responses = np.array([0, 1, 0, 1, 0])
    expected = np.array(
        [[0, 2, 1], [1, 2, 3], [0, 1, 2], [0, 1, 2], [3, 1, 2]])
    assert np.all(unify_triplet_order(triplets, responses) == expected)


def test_normal_triplets_reduce():
    triplets = np.array(
        [[0, 1, 2], [1, 2, 3], [0, 2, 1], [0, 1, 2], [3, 2, 1]])
    responses = np.array([0, 1, 1, 1, 1])
    assert np.all(reduce_triplets(triplets, responses) == np.array(
        [[0, 2, 1], [1, 2, 3], [3, 2, 1]]))
    assert np.all(reduce_triplets(triplets) == np.array(
        [[0, 1, 2], [1, 2, 3], [3, 2, 1]]))
    assert np.all(reduce_triplets(triplets, np.array([1, 1, 1, 1, 1])) == np.array(
        [[0, 1, 2], [1, 2, 3], [3, 2, 1]]))
