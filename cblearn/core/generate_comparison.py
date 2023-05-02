import itertools

import numpy as np
from sklearn.utils import check_random_state


def all_index_tuples(n_objects: int, n_col: int) -> np.ndarray:
    """ Make all index tuples for a number of objects.

    Args:
        n_objects: Number of objects to represent in triplets
        n_col: Number of columns in the output comparisons.
        monotonic: Generate only triplets (i, j, k), such that j < i < k.
    Returns:
        Numpy array (n_triplets, n_col).
    """
    indices = np.arange(n_objects)
    tuple_iter = itertools.chain.from_iterable(itertools.combinations(indices, n_col))
    tuples = np.fromiter(tuple_iter, int).reshape(-1, n_col)
    return tuples


def uniform_index_tuples(n_object: int, n_comps: int, n_col: int, random_state: None | int | np.random.RandomState = None):
    """ Generate random sets of unique integers with Floyd's algorithm.

    This algorithm is both fast and memory efficient such that it scales
    to large datasets.
    Floyd's sampling algorithm assures unbiased insertion probability
    for randomly generated sets.

    A naive implementation would sample rows from all unique combinations of
    the integers 0 to n_objects-1, causing extensive memory usage by the
    number of combinations (n_objects choose n_col).
    In contrast, the memory and runtime usage of this algorithm is linear
    in the number of rows and columns, and constant in the number of objects.

    Args:
        n_object: Number of objects to sample from.
        n_comps: Number of comparisons to sample.
        n_col: Number of columns in the output comparisons.
        random_state: Seed for random sampling.

    Returns:
        comparisons: array of shape (n_comps, n_col) with uniform integers

    References
    ----------
    Bentley, J., & Floyd, B.(1987). Programming pearls: A sample of brilliance.
    Communications of the ACM, 30(9), 754–757. https://doi.org/10.1145/30401.315746
    """
    # Algorithm is explained on https://math.stackexchange.com/q/178690
    query = np.zeros((n_comps, n_col), dtype=np.uint)
    rng = check_random_state(random_state)

    query[:, 0] = rng.randint(0, n_object - n_col + 1, n_comps)
    for i in range(1, n_col):
        j = n_object - n_col + i
        t = rng.randint(0, j + 1, n_comps)
        t_is_new = (query[:, :i] != t.reshape(-1, 1)).all(axis=1)
        query[:, i] = np.where(t_is_new, t, j)

    # Floyd's algorithm does not guarantee equal probability
    # in the order of insertion, so we randomize the column order.
    random_column_order = rng.rand(*query.shape).argsort(axis=1)
    return np.take_along_axis(query, random_column_order, axis=1)