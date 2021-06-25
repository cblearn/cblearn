""" Functions in this file return sampled triplets with answers, based on an artificial embedding and noise.

    Usually they combine functions from _triplet_indices and _triplet_answers
    and are used as a high-level interface to create artificial datasets.
"""
import numpy as np

from typing import Union

from ._triplet_indices import make_all_triplet_indices
from ._triplet_indices import make_random_triplet_indices
from ._triplet_response import noisy_triplet_response
from .. import utils


def make_all_triplets(embedding: np.ndarray, result_format: str, monotonic: bool = False, **kwargs):
    """ Make all possible triplets with answers for the provided embedding or distances.

        Args:
            embedding: Object coordinates or distance matrix
            monotonic: Only triplets (j, i, k), such that j < i < k.
            random_state: Seed for noisy answers
            kwargs: Additional arguments passed to :func:`cblearn.datasets.noisy_triplet_answers`
        Returns:
            The triplets and answers, based on format. See :func:`cblearn.utils.check_triplets`.
    """
    triplets = make_all_triplet_indices(len(embedding), monotonic)
    return noisy_triplet_response(triplets, embedding, result_format=result_format, **kwargs)


def make_random_triplets(embedding: np.ndarray, result_format: str, size: Union[int, float] = 1.,
                         random_state: Union[None, int, np.random.RandomState] = None,
                         repeat: bool = True, monotonic: bool = False, make_all: int = 10000, **kwargs
                         ) -> utils.Query:
    """ Make random triplets with answers for the provided embedding or distances.

        >>> triplets, answers = make_random_triplets(np.random.rand(12, 2), size=1000, result_format='list-boolean')
        >>> answers.shape, np.unique(answers).tolist()
        ((1000,), [False,  True])
        >>> triplets.shape, np.unique(triplets).tolist()
        ((1000, 3), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        Args:
            embedding: Object coordinates (n_objects, n_components) or distance matrix (n_objects, n_objects).
            result_format: Result format
            size: Either absolute or relative number of triplets to generate.
            repeat: Sample triplet indices with repetitions
            monotonic: Sample triplets (j, i, k), such that j < i < k.
            make_all: Choose from all triplets instead of iterative sampling,
                      if the difference between all triplets to the requested number is smaller than this value.
            random_state: Seed for triplet sampling and noisy answers
            kwargs: Additional arguments passed to :func:`cblearn.datasets.noisy_triplet_answers`
        Returns:
            The triplets and answers, based on format. See :func:`cblearn.utils.check_triplets`.
    """
    triplets = make_random_triplet_indices(len(embedding), size, random_state, repeat, monotonic, make_all)
    return noisy_triplet_response(triplets, embedding, result_format=result_format, random_state=random_state, **kwargs)
