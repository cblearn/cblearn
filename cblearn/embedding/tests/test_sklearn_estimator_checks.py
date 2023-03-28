""" Run estimator checks from scikit-learn.

Scikit-learn has a test suite for custom estimators.
They test for common mistakes and compatibility issues.

Unfortunately, they sample featurized input data.
This is why we skip most tests and, to run some,
wrap the ordinal embedding estimators below
to transform featurized input to triplets.

Ultimately, the goal is to replace the skipped tests and
wrapped estimators with a custom test suite, mimicking
the sklearn tests but with triplets as input.
"""
from contextlib import contextmanager

import pytest
import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks

from cblearn.embedding import SOE, MLDS, STE, TSTE, CKL, GNMDS
from cblearn.datasets import make_random_triplets


# Add new estimators here:
ALL_TRIPLET_EMBEDDING_ESTIMATORS = [SOE(), MLDS(), STE(), TSTE(), CKL(), GNMDS()]


def _features_to_triplets(X, y=None):
    """ Guess if input are features and sample triplets then. """
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, int) or (X.shape[1] not in [3, 4]):
        if y is None:
            new_X = make_random_triplets(X, size=len(X), result_format='list-order', random_state=1)
            return new_X
        else:
            new_X, new_y = make_random_triplets(X, size=len(X), result_format='list-count', random_state=1)
            new_y = new_y[:len(y)]  # Make sure new_y has same length as y
            return new_X, new_y
    else:
        if y is None:
            return X
        else:
            return X, y


@contextmanager
def wrap_triplet_estimator(estimator):
    """ Replace feature input estimator input by triplets in context.

    Wraps fit and predict methods on context enter.
    After context exit, original methods are restored.
    """
    orig_fit = estimator.__class__.fit
    orig_fit_transform = estimator.__class__.fit_transform
    methods = ['transform', 'predict', 'predict_proba', 'decision_function']
    orig_methods = {m: getattr(estimator.__class__, m) for m in methods
                    if hasattr(estimator.__class__, m)}

    def new_fit(e, X, y=None):
        if y is None:
            X = _features_to_triplets(X)
        else:
            X, y = _features_to_triplets(X, y)
        return orig_fit(e, X, y)

    estimator.__class__.fit = new_fit
    estimator.__class__.fit_transform = lambda estimator, X, y=None: new_fit(estimator, X, y).transform(X)
    for m in orig_methods:
        setattr(estimator.__class__, m, lambda self, X: orig_methods[m](self, _features_to_triplets(X)))

    yield estimator

    # Context exit
    estimator.__class__.fit = orig_fit
    estimator.__class__.fit_transform = orig_fit_transform
    for m in orig_methods:
        setattr(estimator.__class__, m, orig_methods[m])


SKIP_CHECKS = [
    'check_estimators_nan_inf',
    'check_estimator_sparse_data',
    'check_pipeline_consistency',
    'check_methods_subset_invariance',
    'check_transformer_general',
    'check_transformer_data_not_an_array',
    'check_n_features_in',
    'check_fit2d_1sample',
    'check_fit2d_predict1d',
    'check_fit_score_takes_y',
    'check_estimators_empty_data_messages',
    'check_methods_sample_order_invariance',
    'check_classifiers_regression_target',
    'check_data_conversion_warning',
    'check_decision_proba_consistency',
    'check_supervised_y_2d',
    'check_supervised_y_no_nan',
    'check_classifiers_train',
    'check_classifiers_one_label',
    'check_classifier_data_not_an_array',
    'check_classifiers_classes',
]


@pytest.mark.sklearn
@parametrize_with_checks(
    ALL_TRIPLET_EMBEDDING_ESTIMATORS
)
def test_all_estimators(estimator, check):
    if check.func.__name__ in SKIP_CHECKS:
        pytest.skip("cblearn ordinal embedding estimator's are not fully compatible to sklearn estimators.")

    with wrap_triplet_estimator(estimator) as wrapped_estimator:
        check(wrapped_estimator)
