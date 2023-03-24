""" Function in this file judge triplets, based on ground-truth embedding and possible noise patterns. """
import enum
from typing import Dict, Callable, Optional, Union

from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise
import numpy as np
from numpy.typing import ArrayLike

import cblearn as cbl
from cblearn import utils


class NoiseTarget(enum.Enum):
    POINTS = 'points'
    DIFFERENCES = 'differences'


class Distance(enum.Enum):
    EUCLIDEAN = 'euclidean'
    PRECOMPUTED = 'precomputed'


def noisy_triplet_response(triplets: cbl.Comparison, embedding: ArrayLike, result_format: Optional[str] = None,
                           noise: Union[None, str, Callable] = None, noise_options: Dict = {},
                           noise_target: Union[str, NoiseTarget] = 'differences',
                           random_state: Union[None, int, np.random.RandomState] = None,
                           distance: Union[str, Distance] = 'euclidean'
                           ) -> ArrayLike:
    """ Triplet response for an embedding with noise.

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components) or distance matrix (n_objects, n_objects)
        result_format: Format of the result. If none, keeps input format.
        noise: Noise distribution.
               Can be the name of a distribution function from :class:`numpy.random.RandomState`
               or a function accepting the same arguments.
               If None, no noise will be applied.
        noise_options: Additional arguments passed to the noise function as keyword arguments.
        noise_target: 'points' if noise should be added to triplet coordinates or
                      'differences' if noise should be added to distance difference.
        random_state: State or seed for noise sampling.
        distance: {'euclidean', 'precomputed'}. Specifies distance metrix between embedding points
                  or if distances are passed directly as distance matrix.
    Returns:
        Response in format as defined by response_format,
        either numpy array (n_triplets,) or sparse matrix

        If return_indices is True, a tuple of indices and responses can be returned


    >>> from cblearn.datasets import noisy_triplet_response
    >>> triplets = [[0, 1, 2], [1, 2, 3]]
    >>> embedding = [[0.1], [0.5], [0.9], [1.]]
    >>> noisy_triplet_response(triplets, embedding, result_format='list-count')[1]
    array([1, 1])

    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> distances = euclidean_distances(embedding)
    >>> print(distances.shape)
    (4, 4)
    >>> noisy_triplet_response(triplets, distances, result_format='list-count', distance='precomputed')[1]
    array([1, 1])
    """
    noise_target = NoiseTarget(noise_target)
    distance = Distance(distance)
    quads: np.ndarray = cbl.check_triplets(triplets, return_y=False, canonical=False)
    embedding = check_array(embedding)
    if isinstance(noise, str):
        random_state = check_random_state(random_state)
        noise_fun: Callable = getattr(random_state, noise)
    elif callable(noise):
        noise_fun = noise

    if distance is Distance.EUCLIDEAN:
        X_quads = embedding[quads.ravel()].reshape(-1, quads.shape[1], embedding.shape[1])
        if noise is not None and noise_target is NoiseTarget.POINTS:
            X_quads += noise_fun(size=X_quads.shape, **noise_options)
        near_distance = pairwise.paired_euclidean_distances(X_quads[:, 0], X_quads[:, 1])
        far_distance = pairwise.paired_euclidean_distances(X_quads[:, 2], X_quads[:, 0])
    elif distance is Distance.PRECOMPUTED:
        if noise is not None and noise_target is NoiseTarget.POINTS:
            raise ValueError("Applying noise on points is not possible for precomputed distances.")
        near_distance = embedding[quads[:, 0], quads[:, 1]]
        far_distance = embedding[quads[:, 2], quads[:, 0]]

    differences = far_distance - near_distance
    if noise is not None and noise_target is NoiseTarget.DIFFERENCES:
        differences += noise_fun(size=differences.shape, **noise_options)

    return utils.check_query_response(quads, np.sign(differences), result_format=result_format)


def triplet_response(triplets: cbl.Comparison, embedding: ArrayLike, result_format: Optional[str] = None,
                     distance: Union[str, Distance] = 'euclidean') -> ArrayLike:
    """ Triplet responses for an embedding.

    The default assumes Euclidean distances between embedding points.

    >>> triplets = [[1, 0, 2], [1, 2, 0]]
    >>> points = [[0], [4], [5]]
    >>> triplets, response = triplet_response(triplets, points, result_format='list-boolean')
    >>> response
    array([False, False])

    To use alternative distance metrics, you can pass precomputed distances instead of an embedding.

    >>> from sklearn.metrics import pairwise
    >>> distances = pairwise.manhattan_distances(points)
    >>> triplets, response = triplet_response(triplets, distances, result_format='list-boolean', distance='precomputed')
    >>> response
    array([False, False])

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components)
        result_format: Format of the result. If none, keeps input format.
        distance: {'euclidean', 'precomputed'}. Specifies distance metrix between embedding points
                  or if distances are passed directly as distance matrix.
    Returns:
        Responses in format as defined by response_format
        either numpy array (n_triplets,) or sparse matrix

        If return_indices is True, a tuple of indices and responses can be returned
    """
    return noisy_triplet_response(triplets, embedding, noise=None, distance=distance, result_format=result_format)
