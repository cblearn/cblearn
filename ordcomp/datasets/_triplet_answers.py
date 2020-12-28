""" Function in this file judge triplets, based on ground-truth embedding and possible noise patterns. """
import enum
from typing import Dict, Callable, Optional, Union

from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise
import numpy as np

from ordcomp import utils


class NoiseTarget(enum.Enum):
    POINTS = 'points'
    DIFFERENCES = 'differences'


class Distance(enum.Enum):
    EUCLIDEAN = 'euclidean'
    PRECOMPUTED = 'precomputed'


def noisy_triplet_answers(triplets: utils.Triplets, embedding: np.ndarray,
                          question_format: Optional[str] = None,
                          answer_format: Optional[str] = None,
                          noise: Union[None, str, Callable] = None,
                          noise_options: Dict = {}, noise_target: Union[str, NoiseTarget] = 'differences',
                          random_state: Union[None, int, np.random.RandomState] = None,
                          distance: Union[str, Distance] = 'euclidean'
                          ) -> utils.TripletAnswers:
    """ Triplet answers for an embedding with noise.

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components) or distance matrix (n_objects, n_objects)
        question_format: Format of the triplet questions. If none, keeps input format.
        answer_format: Triplet format of answers.
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
        Answers in format as defined by answer_format,
        either numpy array (n_triplets,) or sparse matrix

        If return_indices is True, a tuple of indices and answers can be returned
    """
    noise_target = NoiseTarget(noise_target)
    distance = Distance(distance)
    input_question_format, input_answer_format = utils.triplet_format(triplets)
    question_format = utils.QuestionFormat(question_format or input_question_format)
    answer_format = utils.AnswerFormat(answer_format or input_answer_format)

    triplets: np.ndarray = utils.check_triplet_questions(triplets, format=utils.QuestionFormat.LIST)
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

    return utils.check_triplet_answers(triplets, answers=(differences < 0), question_format=question_format,
                                       answer_format=answer_format, sort_others=False)


def triplet_answers(triplets: utils.Triplets, embedding: np.ndarray,
                    question_format: Optional[str] = None, answer_format: Optional[str] = None,
                    distance: Union[str, Distance] = 'euclidean') -> utils.TripletAnswers:
    """ Triplet answers for an embedding.

    The default assumes Euclidean distances between embedding points.

    >>> triplets = [[1, 0, 2], [1, 2, 0]]
    >>> points = [[0], [4], [5]]
    >>> triplets, answers = triplet_answers(triplets, points, answer_format='boolean')
    >>> answers
    array([False,  True])

    To use alternative distance metrics, you can pass precomputed distances instead of an embedding.

    >>> from sklearn.metrics import pairwise
    >>> distances = pairwise.manhattan_distances(points)
    >>> triplets, answers = triplet_answers(triplets, distances, answer_format='boolean', distance='precomputed')
    >>> answers
    array([False,  True])

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components)
        question_format: Format of the triplet questions. If none, keeps input format.
        answer_format: Triplet format of answers. If none, keeps input format.
        distance: {'euclidean', 'precomputed'}. Specifies distance metrix between embedding points
                  or if distances are passed directly as distance matrix.
    Returns:
        Answers in format as defined by answer_format
        either numpy array (n_triplets,) or sparse matrix

        If return_indices is True, a tuple of indices and answers can be returned
    """
    return noisy_triplet_answers(triplets, embedding, noise=None,
                                 question_format=question_format, answer_format=answer_format, distance=distance)
