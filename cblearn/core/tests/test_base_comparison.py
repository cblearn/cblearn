import sparse
import numpy as np
import pytest

from cblearn import issparse
from cblearn import assparse, asdense
from cblearn import canonical_X_y


triplets_numeric_undecided = [[0, 1, 2],
                              [0, 1, 2],
                              [0, 1, 2],
                              [3, 0, 2],
                              [1, 2, 2]]
answers_numeric_undecided = [1, 1, 1, -1, 0]

triplets_explicit = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [3, 0, 2]]
answers_numeric = [1, 1, 1, -1]
answers_binary = [True, True, True, False]
triplets_ordered = [[0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [3, 2, 0]]
triplets_spmatrix = sparse.COO(np.transpose(triplets_explicit), answers_numeric, shape=(4, 4, 4))


def test_issparse():
    assert issparse(triplets_spmatrix)
    assert not issparse(triplets_spmatrix, answers_binary)
    assert not issparse(triplets_explicit, answers_binary)

    
def test_assparse():
    assert (assparse(triplets_spmatrix) == triplets_spmatrix).all()
    assert (assparse(triplets_spmatrix.reshape((4, -1)).to_scipy_sparse()) == triplets_spmatrix).all()
    assert (assparse(triplets_explicit, answers_numeric) == triplets_spmatrix).all()

    with pytest.raises(ValueError):
        assparse(triplets_ordered)
    with pytest.raises(ValueError):
        assparse(triplets_explicit, answers_binary) 


def test_asdense():
    X, y = asdense(triplets_spmatrix)
    np.testing.assert_allclose(triplets_explicit, X)
    np.testing.assert_allclose(answers_numeric, y)

    X, y = asdense(triplets_explicit, answers_numeric)
    np.testing.assert_allclose(triplets_explicit, X)
    np.testing.assert_allclose(answers_numeric, y)

    X = asdense(triplets_ordered)
    np.testing.assert_allclose(triplets_ordered, X)
       
       
def test_canonical_X_y():
    X_base = np.array([[1, 2, 3, 4],
                       [2, 1, 4, 3],
                       [4, 3, 2, 1]])

    X, y = canonical_X_y(X_base, np.array([0, 1, 2]))
    np.testing.assert_equal(X, [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    np.testing.assert_equal(y, [0, 0, 1])

    X, y = canonical_X_y(X_base.reshape(-1, 2, 2), np.array([0, 1, 0]))
    np.testing.assert_equal(X.reshape(-1, 4), [[1, 2, 3, 4], [2, 1, 4, 3], [2, 1, 4, 3]])
    np.testing.assert_equal(y, [0, 1, 1])
