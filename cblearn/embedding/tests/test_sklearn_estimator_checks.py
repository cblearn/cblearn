""" Run estimator checks from scikit-learn.

Scikit-learn has a test suite for custom estimators.
They test for common mistakes and compatibility issues.

Unfortunately, they sample featurized input data.
This is why we wrap the ordinal embedding estimators below
to transform featurized input to triplets.

Some checks are skipped, because cblearn's estimators are not
100% compatible to sklearn's estimators.
While some of the checks could be adapted to our setting,
some cannot work with triplet input.
"""
import warnings

import pytest
import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks
import sklearn.utils.estimator_checks
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.utils._tags import (
    _DEFAULT_TAGS,
    _safe_tags,
)

from cblearn.embedding import SOE, MLDS, STE, TSTE, CKL, GNMDS
from cblearn.embedding import wrapper
from cblearn.datasets import make_random_triplets


# Add new estimators here:
ALL_TRIPLET_EMBEDDING_ESTIMATORS = [SOE(), MLDS(), STE(), TSTE(), CKL(), GNMDS()]



##########
# Monkey patching to transform featurized input to triplets
orig_enforce_estimator_tags_X = sklearn.utils.estimator_checks._enforce_estimator_tags_X
orig_enforce_estimator_tags_y = sklearn.utils.estimator_checks._enforce_estimator_tags_y


def _enforce_estimator_tags_X(estimator, X, kernel=linear_kernel):
    X = orig_enforce_estimator_tags_X(estimator, X, kernel)
    if _safe_tags(estimator, key="triplets"):
        n = X.shape[0]
        if len(X) == 1:  # make_random_triplets expects at least 3 objects
            X = np.r_[X, X, X]
        X = make_random_triplets(X, size=n, result_format='list-order')
    return X


def _enforce_estimator_tags_y(estimator, y):
    y = orig_enforce_estimator_tags_y(estimator, y)
    if _safe_tags(estimator, key="triplets"):
        #y = np.where(y == y.flat[0], 1, -1)
        n = y.shape[0]
        y = np.ones(n)
    return y

sklearn.utils.estimator_checks._enforce_estimator_tags_X = _enforce_estimator_tags_X
sklearn.utils.estimator_checks._enforce_estimator_tags_y = _enforce_estimator_tags_y
###########


def test_enforce_estimator_tags_monkeypatch():
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 1)
    estimator = ALL_TRIPLET_EMBEDDING_ESTIMATORS[0]
    assert _safe_tags(estimator).get('triplets', False)
    new_X = sklearn.utils.estimator_checks._enforce_estimator_tags_X(estimator, X)
    new_y = sklearn.utils.estimator_checks._enforce_estimator_tags_y(estimator, y)

    assert new_X.shape[1] == 3
    assert new_y.shape[0] == new_X.shape[0]
    assert new_X.shape[0] >= X.shape[0]
    assert np.isin(np.unique(new_X), np.arange(10)).all()
    np.testing.assert_equal(np.unique(new_y), [1])


# These tests require a 1-to-1 relationship for X -> .transform(X).
# This will never be true for our estimators (n-to-m).
# The alternative to skipping them here would be the 'non_deterministic' tag.
# This tag, however, would skip more tests than necessary.
SKIP_FOR_TRIPLETS = [
    'check_methods_subset_invariance',
    'check_methods_sample_order_invariance'
]

@pytest.mark.sklearn
@pytest.mark.filterwarnings("ignore:Expects the same X queries")  # Expected in check_fit_idem
@parametrize_with_checks(
    ALL_TRIPLET_EMBEDDING_ESTIMATORS
)
def test_all_estimators(estimator, check):
    tags = _safe_tags(estimator)
    if not (tags.get('triplets') and check.func.__name__ in SKIP_FOR_TRIPLETS):
        check(estimator)


@pytest.mark.parametrize(
    "estimator",
    ALL_TRIPLET_EMBEDDING_ESTIMATORS
)
def test_transform_warns_with_other_X(estimator):
    """ Test if warnings are raised when using different X instances in fit and transform. """
    X = np.random.rand(10, 3)
    X = _enforce_estimator_tags_X(estimator, X)
    estimator.fit(X)
    with warnings.catch_warnings(record=True) as w:
        estimator.transform(X)
        estimator.transform(None)
        assert len(w) == 0, "Expected no warnings"

    other_X = X + 1
    with pytest.warns(UserWarning, match="Expects the same X queries in .fit and .transform"):
        estimator.transform(other_X)

    X += 1
    with pytest.warns(UserWarning, match="Expects the same X queries in .fit and .transform"):
        estimator.transform(X)


@pytest.mark.parametrize(
    'estimator',
    ALL_TRIPLET_EMBEDDING_ESTIMATORS
)
def test_make_pipeline(estimator):
    """ Assure that a pipeline can be constructed with ordinal embedding estimators
    and that the resulting pipeline behaves as running the steps individually.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X_feat, y_clust = make_blobs(n_samples=12, n_features=1, centers=3)
    X_trip, y_trip = make_random_triplets(X_feat, size=100, result_format='list-count')
    kmeans = KMeans(3, random_state=42)
    estimator.random_state = 42
    pipe = make_pipeline(estimator, kmeans)

    y_pred_clust_pipe = pipe.fit_predict(X_trip, y_trip)
    X_emb = estimator.fit_transform(X_trip, y_trip)
    y_pred_clust = kmeans.fit_predict(X_emb)

    np.testing.assert_array_equal(y_pred_clust_pipe, y_pred_clust)
