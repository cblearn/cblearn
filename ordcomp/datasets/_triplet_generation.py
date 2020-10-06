import itertools
from typing import Union

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import sklearn
import scipy

from ..utils import check_size


def _make_all_triplets(n_objects: int, monotonic: bool) -> np.ndarray:
    """ Generate all triplets.

    Args:
        n_objects: Number of objects to represent in triplets
        monotonic: Generate only triplets (i, j, k), such that i < j < k.
    Returns:
        Numpy array (n_triplets, 3) of triplet indices.
        n_triplets can become quite large by
    """
    indices = np.arange(n_objects)
    triplet_iter = itertools.chain.from_iterable(
        itertools.combinations(indices, 3))
    triplets = np.fromiter(triplet_iter, int).reshape(-1, 3)

    if monotonic:
        return triplets
    else:
        return np.r_[triplets[:, [1, 0, 2]], triplets, triplets[:, [2, 0, 1]]]


def make_triplets(n_objects: int, size=1., random_state: Union[None, int, np.random.RandomState] = None,
                  repeat: bool = True, monotonic: bool = False, make_all: int = 10000) -> np.ndarray:
    r""" Generate a uniform triplet sample.

    If (almost) all triplets are requested, chooses directly from all possible triplets.
    Otherwise in an iterative approach candidates for triplets are generated to allow sampling for large ``n_objects``.

    Args:
        n_objects: Number of objects to represent in triplets.
        size: Either absolute or relative number of triplets to generate.
        random_state: Seed for random sampling.
        repeat: Sample with repetitions
        monotonic: Sample triplets (i, j, k), such that i < j < k.
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
        triplets = _make_all_triplets(n_objects, monotonic)
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

    return triplets.astype(np.uint)


def _sort_triplets(X):
    return X[np.lexsort((X[:, 2], X[:, 1], X[:, 0]))]


def choose_triplets(objects, size=None, ordered=False, random_state=None):
    if isinstance(objects, int):
        n_objects = objects
    else:
        n_objects = len(objects)

    n_triplets = scipy.special.comb(n_objects, 3, exact=True)
    if not ordered:
        n_triplets *= 3
    n_samples = check_size(size, n_triplets)
    triplets = _make_all_triplets(n_objects)
    n_triplets = triplets.shape[0]

    if ordered:
        X = triplets[:, [1, 0, 2]]
        if n_samples < n_triplets:
            samples = random_state.choice(n_triplets, size=n_samples, replace=False)
            X = X[samples, :]
    else:
        if n_samples < n_triplets:
            samples = random_state.choice(n_triplets, size=n_samples, replace=False)
            X = X[samples, :]
    return _sort_triplets(X)


def make_noisy_triplets(y, size=None, noise=None, options={}, return_responses=False,
                        clip=False, ordered=False, random_state=None):
    """ Simulate triplet questions and the corresponding noisy response.
    Parameters
    ----------
    y: array-like, shape (n_objects, n_input_dim)
        Input data as the individual object values.
    size: int, float or None, default=None
        The number of returned triplets.
         If float, should be between 0.0 and 1.0 and represent the proportion of the available triplets.
         If int, represents the absolute number of triplets.
         If None, all possible triplets are returned..
    noise: str, callable or None, default=None
        The noise added independently on the y values before triplet comparison.
        If string, the corresponding distribution method of a RandomState instance is chosen. If callable,
        it will be called with named arguments *size* (tuple)

        and is expected to return array-like noise with shape *size*.
    options: dict, default={}
        The additional keyword arguments passed to the *noise* function.
    return_responses: boolean, default=False
        If True, returns ``(triplets, responses)`` instead of only ``triplets`` with meaningful column order.
    clip: boolean, default=False
        If True, clip the noisy object values to the original range.
    ordered: boolean, default=False
        If True, consider only triplets with an pivot index between the others indices.
    random_state: int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used generate noise and subsample triplets.
    Returns
    -------
    T: array-like, shape (n_triplets, 3)
        The triplets, where the column order reflects the simulated response.
        For any row ``i` in `T``, the corresponding object of ``T[i, 0]``
          is closer to the object of ``T[i, 1]`` than to the object of ``T[i, 2]``.
    (T, r): tuple if `return_responses` is True
        r is an boolean array-like with shape (n_triplets, 1) with the response.
        For any ``r[i]`` which is *True*, the corresponding object of ``T[i, 0]``
          is closer to the object of ``T[i, 1]`` than to the object of ``T[i, 2]``.
        If ``r[i]` is *False*, then the corresponding object of `T[i, 0]``
          is closer to the object of ``T[i, 2]`` than to the object of ``T[i, 1]``.
    """
    random_state = check_random_state(random_state)
    triplets = choose_triplets(y, size=size, ordered=ordered, random_state=random_state)
    return judge_triplets(triplets, y, noise, options, return_responses, clip, random_state)


def _add_noise(X, noise, options, clip, random_state):
    if isinstance(noise, str):
        random_state = check_random_state(random_state)
        noise = getattr(random_state, noise)
    X = X + noise(size=X.shape, **options)
    if clip:
        return np.clip(X, *clip)
    else:
        return X


def judge_triplets(triplets, y, noise=None, options={}, return_responses=False,
                   clip=False, random_state=None):
    input_dim = y.shape[1]

    y_triplets = y[triplets.ravel()].reshape(-1, 3 * input_dim)
    if clip is True:
        clip = (y.min(), y.max())
    if noise:
        y_triplets = _add_noise(y_triplets, noise, options, clip, random_state)

    pivot, left, right = (y_triplets[:, 0:input_dim],
                          y_triplets[:, input_dim:(2 * input_dim)],
                          y_triplets[:, (2 * input_dim):])
    responses = np.linalg.norm(pivot - left, axis=1) < np.linalg.norm(pivot - right, axis=1)

    if return_responses:
        return triplets, responses
    else:
        return np.where(np.c_[responses, responses, responses], triplets, triplets[:, [0, 2, 1]])


def noisy_distances(y, noise=None, options={}, clip=False, symmetrize=True, random_state=None, **kwargs):
    random_state = check_random_state(random_state)
    if clip is True:
        clip = (y.min(), y.max())
    if noise and False:
        y1 = _add_noise(y, noise, options, clip, random_state)
        y2 = _add_noise(y, noise, options, clip, random_state)
    else:
        y1, y2 = y, y
    distances = sklearn.metric.pairwise.pairwise_distances(y1, y2, **kwargs)
    if symmetrize:
        distances = np.triu(distances, k=0) + np.triu(distances, k=1).T
    return distances


class ChoiceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, size=1, replace=True, random_state=None):
        self.size = size
        self.replace = replace
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        n_items = X.shape[0]
        size = check_size(self.size, n_items)

        self.indices_ = random_state.choice(n_items, size, self.replace)
        return self

    def transform(self, X, y=None):
        return X[self.indices_]
