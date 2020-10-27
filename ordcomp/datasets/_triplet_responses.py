""" Function in this file judge triplets, based on ground-truth embedding and possible noise patterns. """
import enum
from typing import Dict, Callable, Union

from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise
import numpy as np

from ordcomp import utils


class NoiseTarget(enum.Enum):
    POINTS = 'points'
    DIFFERENCES = 'differences'


def noisy_triplet_responses(triplets: utils.IndexTriplets, embedding: np.ndarray,
                            noise: Union[None, str, Callable] = None,
                            noise_options: Dict = {}, noise_target: str = 'points',
                            random_state: Union[None, int, np.random.RandomState] = None) -> np.ndarray:
    """ Triplet responses for an embedding with noise.

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components)
        noise: Noise distribution.
               Can be the name of a distribution function from :class:`numpy.random.RandomState`
               or a function accepting the same arguments.
               If None, no noise will be applied.
        noise_options: Additional arguments passed to the noise function as keyword arguments.
        noise_target: 'points' if noise should be added to triplet coordinates or
                      'differences' if noise should be added to distance difference.
        random_state: State or seed for noise sampling.
    Returns:
        Numpy array of boolean responses, (n_triplets,)
    """
    noise_target = NoiseTarget(noise_target)
    triplets: np.ndarray = utils.check_triplets(triplets, format='array', response_type='implicit')
    embedding = check_array(embedding)
    input_dim = embedding.shape[1]

    y_triplets = embedding[triplets.ravel()].reshape(-1, 3 * input_dim)
    if isinstance(noise, str):
        random_state = check_random_state(random_state)
        noise_fun: Callable = getattr(random_state, noise)
    elif callable(noise):
        noise_fun = noise
    if noise is not None and NoiseTarget is NoiseTarget.POINTS:
        y_triplets += noise_fun(size=y_triplets.shape, **noise_options)

    pivot = y_triplets[:, 0:input_dim]
    differences = (pairwise.paired_euclidean_distances(pivot, y_triplets[:, input_dim:(2 * input_dim)])
                   - pairwise.paired_euclidean_distances(pivot, y_triplets[:, (2 * input_dim):]))
    if noise is not None and noise_target is NoiseTarget.DIFFERENCES:
        differences += noise_fun(size=differences.shape, **noise_options)
    return differences < 0


def triplet_responses(triplets: utils.IndexTriplets, embedding: np.ndarray) -> np.ndarray:
    """ Triplet responses for an embedding.

    >>> triplet_responses([[1, 0, 2], [1, 2, 0]], [[0], [4], [5]])
    array([False,  True])

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components)
    Returns:
        Numpy array of boolean responses, (n_triplets,)
    """
    return noisy_triplet_responses(triplets, embedding, noise=None)
