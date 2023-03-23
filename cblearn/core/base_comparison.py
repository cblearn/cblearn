from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sparse import SparseArray, COO, as_coo
from scipy.sparse import spmatrix
from sklearn.utils import check_X_y, check_array

SparseComparison = SparseArray | spmatrix
Comparison = ArrayLike | SparseComparison


def issparse(X: Comparison, y: Optional[ArrayLike] = None) -> bool:
    """ Check if X and y (optional) are sparse arrays.

    Args:
        X: array-like or sparse matrix. Input data.
        y: array-like, default=None. Targets.

    Returns:
        sparse: `True` if X is a sparse array or matrix,
                `False` otherwise.
    """
    return isinstance(X, SparseComparison) and y is None


def _unroll_X_y(X: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """ Repeat entries, whose responses are multiple of -1 or 1. """
    frequency = np.maximum(1, np.abs(y))  # max to not loose "undecided" (0) responses
    if np.all(frequency == 1):
        return X, y
    else:
        unrolled_y = np.repeat(y.clip(-1, 1), frequency, axis=0)
        return np.repeat(X, frequency, axis=0), unrolled_y


def asdense(X: Comparison, y: Optional[ArrayLike] = None,
            multi_output: bool = False, min_features: int = 3)-> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """ Convert X and y to arrays.

    Args:
        X: array-like or sparse matrix. Input data.
        y: array-like or sparse matrix, default=None. Targets.
        multi_output: Whether to allow multiple targets.
        min_features: Minimum number of features. 

    Returns:
        (comparisons, responses)
    """
    if issparse(X, y):
        X = as_coo(X)
        X, y = X.coords.T, X.data
        X, y = _unroll_X_y(X, y)
    if y is None:
        return check_array(X, dtype=int, ensure_2d=True, ensure_min_features=min_features), None
    else:
        return check_X_y(X, y, dtype=int, 
                         ensure_2d=True,
                         multi_output=multi_output,
                         ensure_min_features=min_features)


def assparse(X: Comparison, y: Optional[ArrayLike] = None) -> COO:
    """ Convert X and y to a sparse array.

    Args:
        X: array-like or sparse matrix. Input data.
        y: array-like or sparse matrix, default=None. Targets in {-1, 1}.
            Must be provided for dense input as it is not possible to infer.
            

    Returns:
        sparse_comparisons
    """
    if not issparse(X, y):
        X, y = check_X_y(X, y, dtype=int)
        if y is None:
            raise ValueError("y must be provided for dense input.")
        n_dim = X.shape[1]
        n_objects = int(X.max() + 1)
        shape = n_dim * (n_objects,)
        if not np.isin(np.unique(y.astype(int)), [-1, 1]).all():
            raise ValueError("y must be {-1, 1} to avoid ambiguity.")
        X, y = COO(X.T, y, shape=shape), None
    elif not isinstance(X, COO):
        X = as_coo(X)
        if len(X.shape) == 2:
            # 2d scipy.sparse to multi-dim sparse
            n_dim = int(np.ceil(np.log(np.product(X.shape)) / np.log(X.shape[0])))
            X = X.reshape(n_dim * (X.shape[0],) )
    elif not issparse(X, y):
        raise ValueError(f"X must be from {SparseComparison}")  

    # force square matrix
    if not all(n == X.shape[0] for n in X.shape[1:]):  
        X = X.reshape(len(X.shape) * (max(X.shape),))

    return X


def canonical_X_y(X: ArrayLike, y: Optional[ArrayLike] = None, axis=1) -> tuple[ArrayLike, ArrayLike] | ArrayLike:
    ind = np.argsort(X, axis=axis)
    X_sorted = np.take_along_axis(X, ind, axis=axis)
    if y is None:
        return X_sorted
    else:
        shape = len(ind.shape) * [1,]
        shape[0] = ind.shape[0]
        shape[axis] = -1
        indind = np.argsort(ind, axis=axis)
        y_sorted = np.take_along_axis(indind, y.reshape(shape), axis=axis)[..., 0].reshape(y.shape)
        return X_sorted, y_sorted