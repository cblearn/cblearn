from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from . import Comparison, canonical_X_y, asdense, assparse, issparse


def check_quadruplets(X: Comparison, y: Optional[ArrayLike] = None,
                      return_y=True,
                      sparse=False,
                      canonical=True) -> Comparison | tuple[Comparison, ArrayLike]:
    """ Check comparisons in the quadruplet format (a, b) vs (c, d)."""
    X, y = asdense(X, y, multi_output=False)
    if X.shape[1] == 3:
        X = X[:, [1, 0, 0, 2]]
    elif X.shape[1] != 4:
        raise ValueError("X must have 3 or 4 columns.")

    if y is not None:
        y = y.astype(int)
        if not np.isin(y.ravel(), [-1, 1]).all():
            raise ValueError(f"y must contain {{-1, 1}}, got {set(np.unique(y))}")
        y = (y > 0).astype(int)  # convert to {0, 1} for canonical_X_y [-1 -> 0, 1 -> 1]

    if return_y and y is None:
        y = np.ones(X.shape[0], dtype=int)

    if canonical:
        # first sort within both pairs
        X = canonical_X_y(X.reshape(-1, 2, 2), axis=2).reshape(-1, 4)
        if return_y:
            if y is None:
                raise ValueError("canonical=True requires y to be provided.")
            # then sort the pairs
            is_sorted = (np.argsort(X.reshape(-1, 2, 2), axis=1)[:, 0, :] == 0).all(axis=1)
            X = np.where(np.c_[is_sorted, is_sorted, is_sorted, is_sorted], X, X[:, [2, 3, 0, 1]])
            y = np.where(is_sorted, y, 1 - y)

    if not return_y and y is not None:
        mask = y.ravel()
        X, y = np.where(np.c_[mask, mask, mask, mask], X, X[:, [2, 3, 0, 1]]), None

    if y is not None:
        y = np.array([-1, 1])[y]  # convert back to {-1, 1}
    if sparse:
        X, y = assparse(X, y), None

    if y is None:
        return X
    else:
        return X, y


def check_triplets(X: Comparison, y: Optional[ArrayLike] = None,
                   return_y=True,
                   sparse=False,
                   canonical=True) -> Comparison | tuple[ArrayLike, ArrayLike]:
    """ Check comparisons in the triplet format (a, b, c) where a is the pivot."""
    X, y = asdense(X, y, multi_output=False)
    if X.shape[1] != 3:
        raise ValueError("X must have 3 columns.")

    if y is not None:
        y = y.astype(int)
        if not np.isin(y.ravel(), [-1, 1]).all():
            raise ValueError(f"y must contain {{-1, 1}}, got {set(np.unique(y))}")
        y = (y > 0).astype(int)  # convert to {0, 1} for canonical_X_y

    if return_y and y is None:
        y = np.ones(X.shape[0], dtype=int)

    if canonical and return_y:
        if y is None:
            raise ValueError("canonical=True requires y to be provided.")
        X = X.copy()
        X[:, 1:], y = canonical_X_y(X[:, 1:], y)

    if not return_y and y is not None:
        mask = y.ravel()
        X, y = np.where(np.c_[mask, mask, mask], X, X[:, [0, 2, 1]]), None

    if y is not None:
        y = np.array([-1, 1])[y]  # convert back to {-1, 1}
    if sparse:
        X, y = assparse(X, y), None

    if y is None:
        return X
    else:
        return X, y


def check_pivot_comparisons(X: Comparison, y: Optional[ArrayLike] = None,
                            select: int = None, return_y=True, canonical=True) -> Comparison | tuple[Comparison, ArrayLike]:
    """ Check comparisons in the pivot format––all comparisons are of the form (a, b) vs (a, c) where a is the pivot.
        The first column is the pivot; responses indicate the columns of selected objects.

        Pivot comparisons are a generalization of the triplet format, where the pivot is always the first column but
        multiple columns can be selected.
        """
    if issparse(X, y):
        raise ValueError("Pivot comparisons are not supported for sparse matrices.")
    X, y = asdense(X, y, multi_output=True)

    if y is not None:
        y_unique = np.unique(y)
        if (y < 0).any() or (y > X.shape[1] - 1).any():
            raise ValueError(f"y must be between 0 and {X.shape[1] - 1}, got {y_unique}")

    if return_y and y is None:
        # order is important: BEFORE canonical
        if select is None:
            raise ValueError("y must be provided if select is not provided.")
        y = np.mgrid[0:len(X), 0:select][1]

    if canonical:
        if y is None:
            raise ValueError("canonical=True requires y to be provided.")
        X = X.copy()
        X[:, 1:], y = canonical_X_y(X[:, 1:], y)

    if not return_y and y is not None:
        # order is important: AFTER canonical
        new_X = X.copy()
        all_rows = np.arange(X.shape[0])
        other_mask = np.ones_like(X, dtype=bool)
        other_mask[:, 0] = False  # pivot column
        for col, col_y in enumerate(y.reshape(X.shape[0], -1).T):
            other_mask[all_rows, col_y + 1] = False
            new_X[all_rows, col + 1] = X[all_rows, col_y + 1]
        new_X[all_rows, (col + 2):] = X[other_mask].reshape(X.shape[0], -1)
        X, y = new_X, None

    if y is None:
        return X
    else:
        return X, y


def check_pairwise_comparisons(X: Comparison, y: Optional[ArrayLike] = None,
                               return_y=True, canonical=True) -> Comparison | tuple[Comparison, ArrayLike]:
    """ Check pairwie comparisons where all entries are compared with each other.
        Responses indicate the column of the selected object.

        Pairwise comparisons are, for example, called "the-odd-one-out" or "the-most-central" in the literature.
    """
    if issparse(X, y):
        raise ValueError("Round-robin comparisons are not supported for sparse matrices.")
    X, y = asdense(X, y, multi_output=False)

    if y is not None:
        y_unique = np.unique(y)
        if (y < 0).any() or (y > X.shape[1]).any():
            raise ValueError(f"y must be between 0 and {X.shape[1]}, got {y_unique}")

    if return_y and y is None:
        # order is important: BEFORE canonical
        y = np.zeros(len(X), dtype=int)

    if canonical:
        if y is None:
            raise ValueError("canonical=True requires y to be provided.")
        X, y = canonical_X_y(X, y)

    if not return_y and y is not None:
        # order is important: AFTER canonical
        all_rows = np.arange(X.shape[0])
        other_mask = np.ones_like(X, dtype=bool)
        other_mask[all_rows, y] = False
        X = np.c_[X[all_rows, y.ravel()], X[other_mask].reshape(X.shape[0], -1)]
        y = None

    if y is None:
        return X
    else:
        return X, y