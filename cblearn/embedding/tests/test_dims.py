""" Tests for the dimensionality validation and estimation utilities. """
import numpy as np
import pytest
from sklearn.base import clone

from cblearn.datasets import make_random_triplets
from cblearn.embedding import CKL, FORTE, GNMDS, LORE, MLDS, OENN, SOE, STE, TSTE
from cblearn.embedding import estimate_dimensionality_cv
from cblearn.embedding._base import check_n_components
from cblearn.embedding._dims import check_test_dimensions


# All estimators with an n_components parameter. Add new estimators here.
ALL_TRIPLET_EMBEDDING_ESTIMATORS = [CKL(), FORTE(), GNMDS(), LORE(backend='scipy', max_iter=10),
                                    MLDS(), OENN(), SOE(n_init=1), STE(), TSTE()]


@pytest.fixture
def triplets():
    random_state = np.random.RandomState(42)
    true_embedding = random_state.rand(12, 2)
    return make_random_triplets(true_embedding, result_format='list-order',
                                size=500, random_state=random_state)


@pytest.mark.parametrize('n_components', [1, 2, 10, np.int64(3)])
def test_check_n_components_accepts_positive_integers(n_components):
    """ A valid dimension keeps its value and is returned as a Python integer. """
    checked = check_n_components(n_components)
    assert checked == int(n_components)
    assert isinstance(checked, int)


@pytest.mark.parametrize('n_components', [0, -1, -10, np.int64(0)])
def test_check_n_components_rejects_non_positive(n_components):
    """ A dimension below 1 describes no embedding space and must be rejected. """
    with pytest.raises(ValueError, match="positive integer"):
        check_n_components(n_components)


@pytest.mark.parametrize('n_components', [2.0, 2.5, np.float64(2.0), '2', None, [2], True])
def test_check_n_components_rejects_non_integers(n_components):
    """ A dimension is a count of axes, so non-integers must be rejected. """
    with pytest.raises(ValueError, match="positive integer"):
        check_n_components(n_components)


def test_check_n_components_message_contains_parameter_name():
    """ The error message names the parameter to fix. """
    with pytest.raises(ValueError, match="test_dimensions"):
        check_n_components(0, name="test_dimensions")


def test_check_test_dimensions_accepts_increasing_dimensions():
    """ Valid dimensions are returned as an array of the same values. """
    np.testing.assert_array_equal(check_test_dimensions([1, 2, 5]), np.array([1, 2, 5]))


@pytest.mark.parametrize('test_dimensions,invalid_index', [([0, 1, 2], 0), ([-1, 0], 0),
                                                           ([1, 2, 2.5], 2), ([1, 0, 3], 1)])
def test_check_test_dimensions_rejects_invalid_dimensions(test_dimensions, invalid_index):
    """ Every tested dimension must be a positive integer,
        and the message names the entry to fix. """
    with pytest.raises(ValueError, match=fr"test_dimensions\[{invalid_index}\]"):
        check_test_dimensions(test_dimensions)


@pytest.mark.parametrize('test_dimensions', [[], [2], [[1, 2], [3, 4]]])
def test_check_test_dimensions_rejects_too_few_dimensions(test_dimensions):
    """ The sequential test compares consecutive dimensions, so at least two are required. """
    with pytest.raises(ValueError, match="at least two dimensions"):
        check_test_dimensions(test_dimensions)


@pytest.mark.parametrize('test_dimensions', [[3, 2, 1], [1, 1, 2], [1, 3, 2]])
def test_check_test_dimensions_rejects_non_increasing(test_dimensions):
    """ The sequential test assumes monotonically increasing dimensions. """
    with pytest.raises(ValueError, match="monotonically increasing"):
        check_test_dimensions(test_dimensions)


@pytest.mark.parametrize('n_components', [0, -1, 2.5])
def test_estimator_fit_rejects_invalid_dimension(triplets, n_components):
    """ Estimators name the invalid parameter, instead of raising the numpy
        reshape error that an invalid n_components produces downstream. """
    with pytest.raises(ValueError, match="n_components"):
        SOE(n_components=n_components, n_init=1).fit(triplets)


def test_estimate_dimensionality_cv_rejects_zero_dimension(triplets):
    """ A dimension list starting at 0, such as range(10), is rejected. """
    with pytest.raises(ValueError, match="test_dimensions"):
        estimate_dimensionality_cv(SOE(n_init=1), triplets, test_dimensions=list(range(10)),
                                   n_splits=2, n_jobs=1, random_state=42)


def test_estimate_dimensionality_cv_recovers_dimension(triplets):
    """ Two-dimensional data is described by a two-dimensional embedding. """
    result = estimate_dimensionality_cv(SOE(n_init=1), triplets, test_dimensions=[1, 2, 3],
                                        n_splits=5, n_jobs=1, random_state=42)
    assert result.estimated_dimension == 2
    np.testing.assert_array_equal(result.dimensions, [1, 2, 3])


@pytest.mark.parametrize('estimator', ALL_TRIPLET_EMBEDDING_ESTIMATORS,
                         ids=[type(e).__name__ for e in ALL_TRIPLET_EMBEDDING_ESTIMATORS])
@pytest.mark.parametrize('n_components', [0, -1, 2.5])
def test_all_estimators_reject_invalid_dimension(triplets, estimator, n_components):
    """ All estimators share the same error for an invalid embedding dimension. """
    estimator = clone(estimator).set_params(n_components=n_components)
    with pytest.raises(ValueError, match="n_components"):
        estimator.fit(triplets)
