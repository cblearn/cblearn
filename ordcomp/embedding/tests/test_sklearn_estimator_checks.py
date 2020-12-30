""" Run estimator checks from scikit-learn.

Scikit-learn has a test suite for custom estimators.
They test for common mistakes and compatibility issues.

Unfortunately, they sample featurized input data.
This is why we wrap the ordinal embedding estimators below
to transform featurized input to triplets.

Some checks are skipped, because ordcomp's estimators are not
100% compatible to sklearn's estimators.
While some of the checks could be adapted to our setting,
some cannot work with triplet input.
"""
from contextlib import contextmanager

import pytest
import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks

from ordcomp.embedding import wrapper
from ordcomp.datasets import make_random_triplets


# Add new estimators here:
ALL_TRIPLET_EMBEDDING_ESTIMATORS = [wrapper.SOE(), wrapper.MLDS()]


def _features_to_triplets(X):
    """ Guess if input are features and sample triplets then. """
    if isinstance(X, np.ndarray) and (
            not np.issubdtype(X.dtype, np.uint) or X.ndim != 2 or X.shape[1] != 3):
        n = X.size
        new_X = make_random_triplets(X, size=n, result_format='list-order', random_state=1)
        print("to triplets", X.shape, X.dtype, new_X.shape, new_X.dtype)
        return new_X
    else:
        print("are triplets", np.asarray(X).shape)
        return X


@contextmanager
def wrap_triplet_estimator(estimator):
    """ Replace feature input estimator input by triplets in context.

    Wraps fit and predict methods on context enter.
    After context exit, original methods are restored.
    """
    orig_fit = estimator.__class__.fit
    orig_predict = estimator.__class__.predict

    estimator.__class__.fit = lambda self, X, y=None: orig_fit(self, _features_to_triplets(X))
    estimator.__class__.predict = lambda self, X: orig_predict(self, _features_to_triplets(X))
    yield estimator

    # Context exit
    estimator.__class__.fit = orig_fit
    estimator.__class__.predict = orig_predict


SKIP_CHECKS = [
    'check_estimators_nan_inf',
    'check_estimator_sparse_data',
    'check_estimators_pickle',
    'check_pipeline_consistency',
    'check_methods_subset_invariance',
    'check_transformer_general',
    'check_transformer_data_not_an_array',
    'check_n_features_in',
    'check_fit2d_1sample',
    'check_fit2d_predict1d',
    'check_fit_score_takes_y',
    'check_estimators_empty_data_messages',
]


@parametrize_with_checks(
    ALL_TRIPLET_EMBEDDING_ESTIMATORS
)
def test_all_estimators(estimator, check):
    if check.func.__name__ in SKIP_CHECKS:
        pytest.skip("Ordcomp ordinal embedding estimator's are not fully compatible to sklearn estimators.")

    with wrap_triplet_estimator(estimator) as wrapped_estimator:
        check(wrapped_estimator)
