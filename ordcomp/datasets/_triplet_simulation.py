""" Functions in this file return sampled triplets with answers, based on an artificial embedding and noise.

    Usually they combine functions from _triplet_indices and _triplet_answers
    and are used as a high-level interface to create artificial datasets.
"""
import numpy as np

from typing import Union, Callable, Dict

from ._triplet_indices import make_random_triplet_indices
from ._triplet_answers import noisy_triplet_answers
from .. import utils


def make_random_triplets(embedding: np.ndarray, size: Union[int, float] = 1., answer_format='boolean',
                         noise: Union[None, str, Callable] = None, noise_options: Dict = {},
                         noise_target: str = 'points', random_state: Union[None, int, np.random.RandomState] = None,
                         repeat: bool = True, monotonic: bool = False, make_all: int = 10000) -> utils.Triplets:
    """ Make triplets with answers.

        >>> triplets, answers = make_random_triplets(np.random.rand(12, 2),size=1000)
        >>> answers.shape, np.unique(answers).tolist()
        ((1000,), [False,  True])
        >>> triplets.shape, np.unique(triplets).tolist()
        ((1000, 3), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        Args:
            embedding: Object coordinates as numpy array, (n_objects, n_components).
            size: Either absolute or relative number of triplets to generate.
            answer_format: Result answer format.
            noise: Noise distribution for answers.
                   Can be the name of a distribution function from :class:`numpy.random.RandomState`
                   or a function accepting the same arguments.
                   If None, no noise will be applied.
            noise_options: Additional arguments passed to the noise function as keyword arguments.
            noise_target: 'points' if noise should be added to triplet coordinates or
                          'differences' if noise should be added to distance difference.
            random_state: State or seed for triplet index and noise sampling.
            repeat: Sample triplet indices with repetitions
            monotonic: Sample triplets (j, i, k), such that j < i < k.
            make_all: Choose from all triplets instead of iterative sampling,
                      if the difference between all triplets to the requested number is smaller than this value.
        Returns:
            The triplets and answers, based on format. See :func:`ordcomp.utils.check_triplets`.
    """
    n_objects, n_dimension = embedding.shape
    triplets = make_random_triplet_indices(n_objects, size, random_state, repeat, monotonic, make_all)
    return noisy_triplet_answers(triplets, embedding, answer_format=answer_format,
                                 noise=noise, noise_options=noise_options,
                                 noise_target=noise_target, random_state=random_state)
