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
from contextlib import contextmanager

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


@pytest.mark.sklearn
@parametrize_with_checks(
    ALL_TRIPLET_EMBEDDING_ESTIMATORS
)
def test_all_estimators(estimator, check):
    check(estimator)
