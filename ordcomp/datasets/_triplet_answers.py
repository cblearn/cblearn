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


def noisy_triplet_answers(triplets: utils.Triplets, embedding: np.ndarray,
                          question_format: Optional[str] = None,
                          answer_format: Optional[str] = None,
                          noise: Union[None, str, Callable] = None,
                          noise_options: Dict = {}, noise_target: str = 'points',
                          random_state: Union[None, int, np.random.RandomState] = None
                          ) -> utils.TripletAnswers:
    """ Triplet answers for an embedding with noise.

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components)
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
    Returns:
        Answers in format as defined by answer_format,
        either numpy array (n_triplets,) or sparse matrix

        If return_indices is True, a tuple of indices and answers can be returned
    """
    noise_target = NoiseTarget(noise_target)
    input_question_format, input_answer_format = utils.triplet_format(triplets)
    question_format = utils.QuestionFormat(question_format or input_question_format)
    answer_format = utils.AnswerFormat(answer_format or input_answer_format)

    triplets: np.ndarray = utils.check_triplet_questions(triplets, format=utils.QuestionFormat.LIST)
    embedding = check_array(embedding)
    input_dim = embedding.shape[1]

    y_triplets = embedding[triplets.ravel()].reshape(-1, 3 * input_dim)
    if isinstance(noise, str):
        random_state = check_random_state(random_state)
        noise_fun: Callable = getattr(random_state, noise)
    elif callable(noise):
        noise_fun = noise
    if noise is not None and noise_target is NoiseTarget.POINTS:
        y_triplets += noise_fun(size=y_triplets.shape, **noise_options)

    pivot = y_triplets[:, 0:input_dim]
    differences = (pairwise.paired_euclidean_distances(pivot, y_triplets[:, input_dim:(2 * input_dim)])
                   - pairwise.paired_euclidean_distances(pivot, y_triplets[:, (2 * input_dim):]))
    if noise is not None and noise_target is NoiseTarget.DIFFERENCES:
        differences += noise_fun(size=differences.shape, **noise_options)

    return utils.check_triplet_answers(triplets, answers=(differences < 0), question_format=question_format,
                                       answer_format=answer_format, sort_others=False)


def triplet_answers(triplets: utils.Triplets, embedding: np.ndarray,
                    question_format: Optional[str] = None, answer_format: Optional[str] = None) -> utils.TripletAnswers:
    """ Triplet answers for an embedding.

    >>> triplets, answers = triplet_answers([[1, 0, 2], [1, 2, 0]], [[0], [4], [5]], answer_format='boolean')
    >>> answers
    array([False,  True])

    Args:
        triplets: Numpy array or sparse matrix of triplet indices
        embedding: Numpy array of object coordinates, (n_objects, n_components)
        question_format: Format of the triplet questions. If none, keeps input format.
        answer_format: Triplet format of answers. If none, keeps input format.
        return_indices: Return triplet indices along with answers
    Returns:
        Answers in format as defined by answer_format
        either numpy array (n_triplets,) or sparse matrix

        If return_indices is True, a tuple of indices and answers can be returned
    """
    return noisy_triplet_answers(triplets, embedding, noise=None,
                                 question_format=question_format, answer_format=answer_format)
