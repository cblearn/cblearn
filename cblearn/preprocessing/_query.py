from itertools import permutations
from typing import Union, Optional

import numpy as np
from sklearn.utils import check_array, check_X_y


def triplets_from_multiselect(X: np.ndarray, select: Union[np.ndarray, int], is_ranked: bool) -> np.ndarray:
    """ Calculate triplets from n-select or n-rank queries.

    The n-select query consists of :math:`k>2` object indices.
    The first index indicates the pivot object. The selected objects
    should be more similar to the pivot, than the other objects.
    The selected objects themself can be ordered in similarity to the pivot (`is_ranked=True`).

    Triplets are a special case of n-select queries, with a single other object.

    Args:
         X: n-select or n-rank query as array (n_query, n_choices)
         select: Integer of first n selected columns or a 2d array (n_query, n_select)
                 of column indices in X (0..n_choices).
         is_ranked: If true, assumes that the selected objects are ordered by their similarity.
    Return:
        triplets: Array of triplet queries
                  (n_query * (2 * (n_choices - n_select - 1) + int(is_ranked)), 3)
    """
    if isinstance(select, int):
        X = check_array(X)
        n_select = select
    else:
        unordered_X, y = check_X_y(X, select, multi_output=True)
        n_select = y.shape[1]
        all_rows = np.arange(unordered_X.shape[0])
        other_mask = np.ones_like(unordered_X, dtype=bool)
        other_mask[:, 0] = False  # pivot column
        X = np.array(unordered_X)
        for col, selected_col in enumerate(y.T):
            other_mask[all_rows, selected_col] = False
            X[all_rows, col + 1] = unordered_X[all_rows, selected_col]
        X[all_rows, (n_select + 1):] = unordered_X[other_mask].reshape(X.shape[0], -1)

    n_trials, n_stimuli = X.shape
    ix_array = np.array([
        [0, ix_select, ix_other]
        for ix_select in range(1, n_select + 1)
        for ix_other in range(ix_select + 1 if is_ranked else n_select + 1, n_stimuli)
    ])
    return np.concatenate(X[:, ix_array])


def triplets_from_oddoneout(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """ Calculates triplets from odd-one-out queries.

    The odd-one-out query consists of k objects, of which one
    is most dissimilar to all others.

    .. Note::
        For this transformation, we are assuming that the objects respect the triangle inequality.
        This might not always be a given and is not checked by this function.

    Args:
        X: Array of odd-one-out queries (n_query, n_choices)
        y: Optional list of indices, that indicate the odd choice per query.
           If omitted, the first entry is assumed to be the odd object.
    Returns:
        triplets: Array of triplet queries (n_query * (n_choices - 2) * (n_choices - 1), 3)
    """
    if y is None:
        y = np.zeros(len(X), dtype=int)

    X, y = check_X_y(X, y)
    mask = np.zeros_like(X, dtype=bool)
    mask[np.arange(X.shape[0]), y] = True
    far = X[mask]
    others = X[~mask].reshape(X.shape[0], X.shape[1] - 1)
    triplets = []
    for other_ix in permutations(np.arange(others.shape[1]), 2):
        triplets.append(np.c_[others[:, other_ix], far])
    return np.row_stack(triplets)


def triplets_from_mostcentral(X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """ Calculates triplets from most-central queries.

    The most-central query consists of k objects, of which one
    is most similar to all others.

    .. Note::
        For this transformation, we are assuming that the objects respect the triangle inequality.
        This might not always be a given and is not checked by this function.

    Args:
        X: Array of most-central queries (n_query, n_choices)
        y: Optional list of indices, that indicate the central choice per query.
           If omitted, the first entry is assumed to be the odd object.
    Returns:
        triplets: Array of triplet queries (n_query * (n_choices - 2) * (n_choices - 1), 3)
    """
    triplets = triplets_from_oddoneout(X, y)
    return triplets[:, [0, 2, 1]]
