from typing import Union, TypeVar

import numpy as np
from sklearn.utils import check_array
from sklearn import metrics
import sparse
import scipy

from cblearn import utils
from .. import datasets


A = TypeVar('A', utils.Answers, np.ndarray)


def triplet_score(true_answers: A, embedding_or_pred_answers: Union[np.ndarray, A], distance=None) -> float:
    """Fraction of violated triplet constraints.

    For all triplets (i, j, k), count R * (||O(j) - O(i)|| - ||O(k) - O(i)||) > 0
    and divide by the number of triplets.

    Args:
        triplets: Triplet constraints either in array or sparse matrix format
        embedding_or_pred_answers: Either object coordinates, shape (n_objects, n_features),
                                    or predicted triplet answers.
    Returns:
        Number between 0 and 1, indicating the fraction of triplet constraints which are violated.
    """
    if not isinstance(true_answers, (tuple, sparse.COO, scipy.sparse.spmatrix)) and np.asarray(true_answers).ndim == 1:
        # Assume only a sequence of answers was passed
        triplets, true_answers = None, true_answers.astype(int)
    else:
        # Assume a complete triplet question+answer was passed
        triplets, true_answers = utils.check_triplet_answers(true_answers, result_format='list-boolean')

    if not isinstance(embedding_or_pred_answers, (tuple, sparse.COO, scipy.sparse.spmatrix)) \
            and np.asarray(embedding_or_pred_answers).ndim == 1:
        # Assume only a sequence of answers was passed
        pred_triplets, pred_answers = None, embedding_or_pred_answers.astype(int)
    elif isinstance(embedding_or_pred_answers, (np.ndarray, list)) and len(embedding_or_pred_answers) != len(triplets):
        # Assume an embedding was passed
        embedding = check_array(embedding_or_pred_answers, ensure_2d=True)
        pred_triplets, pred_answers = datasets.triplet_answers(triplets, embedding, distance=distance,
                                                               result_format='list-boolean')
    else:
        # Assume a complete triplet question+answer was passed
        pred_triplets, pred_answers = utils.check_triplet_answers(embedding_or_pred_answers, result_format='list-boolean')

    if pred_triplets is not None and np.any(triplets != pred_triplets):
        raise ValueError("Expects identical questions for true and predicted triplets.")
    return metrics.accuracy_score(true_answers, pred_answers)


def triplet_error(true_answers: A, embedding_or_pred_answers: Union[np.ndarray, A], distance=None) -> float:
    return 1 - triplet_score(true_answers, embedding_or_pred_answers, distance)

TripletScorer = metrics.make_scorer(lambda y_true, y_pred: triplet_score(y_true, y_pred))
