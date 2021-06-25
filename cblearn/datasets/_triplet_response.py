""" Function in this file judge triplets, based on ground-truth embedding and possible noise patterns. """
import enum
from typing import Dict, Callable, Optional, Union

from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise
import numpy as np

from cblearn import utils


class NoiseTarget(enum.Enum):
    POINTS = 'points'
    DIFFERENCES = 'differences'


class Distance(enum.Enum):
    EUCLIDEAN = 'euclidean'
    PRECOMPUTED = 'precomputed'


def noisy_triplet_response(triplets: utils.Query, embedding: np.ndarray, result_format: Optional[str] = None,
                           noise: Union[None, str, Callable] = None, noise_options: Dict = {},
                           noise_target: Union[str, NoiseTarget] = 'differences',
                           random_state: Union[None, int, np.random.RandomState] = None,
                           distance: Union[str, Distance] = 'euclidean'
                           ) -> utils.Response:
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
    >>> noisy_triplet_response(triplets, embedding, result_format='list-order')
    array([[0, 1, 2],
           [1, 2, 3]], dtype=uint32)

    >>> noisy_triplet_response(triplets, embedding, result_format='list-order',
    ...                       noise='normal', noise_options={'scale': 1}, random_state=42)
    array([[0, 2, 1],
           [1, 2, 3]], dtype=uint32)

    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> distances = euclidean_distances(embedding)
    >>> print(distances.shape)
    (4, 4)
    >>> noisy_triplet_response(triplets, distances, result_format='list-order', distance='precomputed')
    array([[0, 1, 2],
           [1, 2, 3]], dtype=uint32)
    """
    noise_target = NoiseTarget(noise_target)
    distance = Distance(distance)
    result_format = utils.check_format(result_format, triplets, None)
    triplets: np.ndarray = utils.check_query(triplets, result_format=utils.QueryFormat.LIST)
    embedding = check_array(embedding)
    if isinstance(noise, str):
        random_state = check_random_state(random_state)
        noise_fun: Callable = getattr(random_state, noise)
    elif callable(noise):
        noise_fun = noise

    if distance is Distance.EUCLIDEAN:
        y_triplets = embedding[triplets.ravel()].reshape(-1, 3, embedding.shape[1])
        if noise is not None and noise_target is NoiseTarget.POINTS:
            y_triplets += noise_fun(size=y_triplets.shape, **noise_options)
        near_distance = pairwise.paired_euclidean_distances(y_triplets[:, 0], y_triplets[:, 1])
        far_distance = pairwise.paired_euclidean_distances(y_triplets[:, 0], y_triplets[:, 2])
    elif distance is Distance.PRECOMPUTED:
        if noise is not None and noise_target is NoiseTarget.POINTS:
            raise ValueError("Applying noise on points is not possible for precomputed distances.")
        near_distance = embedding[triplets[:, 0], triplets[:, 1]]
        far_distance = embedding[triplets[:, 0], triplets[:, 2]]

    differences = near_distance - far_distance
    if noise is not None and noise_target is NoiseTarget.DIFFERENCES:
        differences += noise_fun(size=differences.shape, **noise_options)

    return utils.check_query_response(triplets, response=(differences < 0), result_format=result_format, standard=False)


def triplet_response(triplets: utils.Query, embedding: np.ndarray, result_format: Optional[str] = None,
                     distance: Union[str, Distance] = 'euclidean') -> utils.Response:
    """ Triplet responses for an embedding.

    The default assumes Euclidean distances between embedding points.

    >>> triplets = [[1, 0, 2], [1, 2, 0]]
    >>> points = [[0], [4], [5]]
    >>> triplets, response = triplet_response(triplets, points, result_format='list-boolean')
    >>> triplets, response
    (array([[1, 0, 2],
           [1, 0, 2]], dtype=uint32), array([False, False]))

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
    return noisy_triplet_response(triplets, embedding, noise=None, result_format=result_format, distance=distance)
