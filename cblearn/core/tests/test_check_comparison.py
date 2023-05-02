import pytest
import numpy as np
import sparse

from cblearn import check_quadruplets, check_triplets, check_pivot_comparisons, check_pairwise_comparisons


X_triplet = [[0, 1, 2],
             [0, 1, 2],
             [0, 1, 2],
             [3, 0, 2]]
X_quad = [[1, 0, 0, 2],
          [1, 0, 0, 2],
          [1, 0, 0, 2],
          [0, 3, 3, 2]]
X_quad_canonical = [[0, 1, 0, 2],
                    [0, 1, 0, 2],
                    [0, 1, 0, 2],
                    [0, 3, 2, 3]]
X_robin_canonical = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [0, 2, 3]]
X_robin_ordered = [[0, 1, 2],
                   [0, 1, 2],
                   [0, 1, 2],
                   [0, 2, 3]]
y_robin_index = [0, 0, 0, 0]
y_numeric = [1, 1, 1, -1]
y_index = [0, 0, 0, 1]
y_binary = [True, True, True, False]
X_triplet_ordered = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [3, 2, 0]]
X_quad_ordered = [[0, 1, 0, 2],
                  [0, 1, 0, 2],
                  [0, 1, 0, 2],
                  [2, 3, 0, 3]]
# WARNING:
# Avoid using np.testing.assert_equal on sparse matrix -> leads to false positives
X_triplet_sparse = sparse.COO(np.transpose(X_triplet), y_numeric, shape=(4, 4, 4))
X_quad_sparse = sparse.COO(np.transpose(X_quad), y_numeric, shape=(4, 4, 4, 4))


def test_check_triplets():
    X, y = check_triplets(X_triplet, y_numeric)
    np.testing.assert_equal(X, X_triplet)
    np.testing.assert_equal(y, y_numeric)

    X, y = check_triplets(X_triplet_ordered)
    np.testing.assert_equal(X, X_triplet)
    np.testing.assert_equal(y, y_numeric)

    X, y = check_triplets(X_triplet_sparse)
    np.testing.assert_equal(X, X_triplet)
    np.testing.assert_equal(y, y_numeric)

    X = check_triplets(X_triplet_ordered, sparse=True)
    np.testing.assert_equal(X, X_triplet_sparse)

    X = check_triplets(X_triplet, y_numeric, return_y=False)
    np.testing.assert_equal(X, X_triplet_ordered)

    with pytest.raises(ValueError):
        # not a triplet
        check_triplets(X_quad)

    with pytest.raises(ValueError):
        # wrong responses
        check_triplets(X_triplet, np.array(y_numeric) + 1)


def test_check_quadruplets():
    X, y = check_quadruplets(X_quad, y_numeric)
    np.testing.assert_equal(X, X_quad_canonical)
    np.testing.assert_equal(y, y_numeric)

    X, y = check_quadruplets(X_quad_ordered)
    np.testing.assert_equal(X, X_quad_canonical)
    np.testing.assert_equal(y, y_numeric)

    X, y = check_quadruplets(X_quad_sparse)
    np.testing.assert_equal(X,
                            [[0, 3, 2, 3], [0, 1, 0, 2], [0, 1, 0, 2], [0, 1, 0, 2]])
    np.testing.assert_equal(y, [-1, 1, 1, 1])

    X = check_quadruplets(X_quad_ordered, sparse=True)
    np.testing.assert_equal(X, X_quad_sparse)

    X = check_quadruplets(X_quad, y_numeric, return_y=False)
    np.testing.assert_equal(X, X_quad_ordered)

    X, y = check_quadruplets(X_triplet, y_numeric)
    np.testing.assert_equal(X, X_quad_canonical)
    np.testing.assert_equal(y, y_numeric)

    with pytest.raises(ValueError):
        # not a triplet
        check_quadruplets(np.array(X_quad)[:, [0, 1, 2, 3, 0]])

    with pytest.raises(ValueError):
        # wrong responses
        check_quadruplets(X_quad, np.array(y_numeric) + 1)


def test_check_pivot_comparisons():
    X, y = check_pivot_comparisons(X_triplet, y_index)
    np.testing.assert_equal(X, X_triplet)
    np.testing.assert_equal(y, y_index)

    X, y = check_pivot_comparisons(X_triplet_ordered, select=1)
    np.testing.assert_equal(X, X_triplet)
    np.testing.assert_equal(y.ravel(), y_index)

    X = check_pivot_comparisons(X_triplet, y_index, return_y=False)
    np.testing.assert_equal(X, X_triplet_ordered)

    with pytest.raises(ValueError):
        # no select or y
        check_pivot_comparisons(X_triplet_ordered)

    with pytest.raises(ValueError):
        # sparse
        check_pivot_comparisons(X_triplet_sparse)

    with pytest.raises(ValueError):
        # invalid responses
        check_pivot_comparisons(X_triplet, np.full_like(y_index, -1))


def test_check_robin_comparisons():
    X, y = check_pairwise_comparisons(X_triplet, y_index)
    np.testing.assert_equal(X, X_robin_canonical)
    np.testing.assert_equal(y, y_robin_index)

    X, y = check_pairwise_comparisons(X_triplet_ordered)
    np.testing.assert_equal(X, X_robin_canonical)
    np.testing.assert_equal(y, [0, 0, 0, 2])

    X = check_pairwise_comparisons(X_triplet, y_index, return_y=False)
    np.testing.assert_equal(X, X_robin_ordered)

    with pytest.raises(ValueError):
        # sparse input
        check_pairwise_comparisons(X_triplet_sparse)

    with pytest.raises(ValueError):
        # invalid responses
        check_pairwise_comparisons(X_triplet, np.full_like(y_index, -1))