""" Functions in this file generate triplet indices without answers.
"""
import itertools
from typing import Union

from sklearn.utils import check_random_state
import numpy as np
import scipy

from ..utils import check_size


def make_all_triplet_indices(n_objects: int, monotonic: bool) -> np.ndarray:
    """ Make all triplet indices for a number of objects.

    Args:
        n_objects: Number of objects to represent in triplets
        monotonic: Generate only triplets (i, j, k), such that j < i < k.
    Returns:
        Numpy array (n_triplets, 3) of triplet indices.
        n_triplets can become quite large by
    """
    indices = np.arange(n_objects)
    triplet_iter = itertools.chain.from_iterable(
        itertools.combinations(indices, 3))
    triplets = np.fromiter(triplet_iter, int).reshape(-1, 3)

    if monotonic:
        return triplets[:, [1, 0, 2]]
    else:
        return np.r_[triplets[:, [1, 0, 2]], triplets, triplets[:, [2, 0, 1]]]


def make_random_triplet_indices(n_objects: int, size: Union[int, float] = 1.,
                                random_state: Union[None, int, np.random.RandomState] = None,
                                repeat: bool = True, monotonic: bool = False, make_all: int = 10000) -> np.ndarray:
    r""" Sample random triplet indices.

    If (almost) all triplets are requested, chooses directly from all possible triplets.
    Otherwise in an iterative approach candidates for triplets are generated to allow sampling for large ``n_objects``.

    >>> triplets = make_random_triplet_indices(n_objects=12, size=1000)
    >>> triplets.shape
    (1000, 3)
    >>> np.unique(triplets)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    Args:
        n_objects: Number of objects to represent in triplets.
        size: Either absolute or relative number of triplets to generate.
        random_state: Seed for random sampling.
        repeat: Sample with repetitions
        monotonic: Sample triplets (i, j, k), such that j < i < k.
        make_all: Choose from all triplets instead of iterative sampling,
                  if the difference between all triplets to the requested number is smaller than this value.
    Returns:
        Numpy array (n_triplets, 3) of triplets.

        n_triplets is determined through size.
        Size of 1. corresponds to all available triplets:

            - If ``monotonic=False`` :math:`\text{n_all_triplets}=3 \dot {\text{n_objects} \choose 3}`.
            - If ``monotonic=True`` :math:`\text{n_all_triplets}={\text{n_objects} \choose 3}`.

    Raises:
        ValueError: If ``repeat = False`` but more ``size`` is larger than all possible triplets for ``n_objects``.
    """
    n_triplets = scipy.special.binom(n_objects, 3)
    if not monotonic:
        n_triplets *= 3

    n_choose = check_size(size, n_triplets)
    random_state = check_random_state(random_state)
    if not repeat and size > n_triplets:
        raise ValueError(f"Cannot sample {n_choose} from {n_triplets} without repetitions.")
    if n_triplets - n_choose < make_all:
        triplets = make_all_triplet_indices(n_objects, monotonic)
        ix = random_state.choice(len(triplets), n_choose, replace=repeat)
        return triplets[ix]

    pairs = np.transpose(np.triu_indices(n_objects, 1))
    triplets = np.empty((0, 3))
    while triplets.shape[0] < n_choose:
        i = random_state.randint(0, n_objects, n_choose)
        jk = pairs[random_state.randint(0, len(pairs), n_choose)]

        if monotonic:
            mask = np.logical_and(jk[:, 0] < i, i < jk[:, 1])
        else:
            mask = np.logical_and(i != jk[:, 0], jk[:, 1] != i)
        triplet_candidates = np.c_[i[mask], jk[mask]]
        triplets = np.r_[triplets, triplet_candidates[:(n_choose - triplets.shape[0])]]
        if not repeat:
            triplets = np.unique(triplets, axis=0)

    return triplets.astype(np.uint32)
