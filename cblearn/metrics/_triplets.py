import numpy as np
from sklearn.utils import check_array
from sklearn import metrics
import sparse
import scipy

from cblearn import utils
from .. import datasets


def query_accuracy(true_response: utils.Answers, pred_response: utils.Answers) -> float:
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
    if not isinstance(true_response, (sparse.COO, scipy.sparse.spmatrix)) and np.asarray(true_response).ndim == 1:
        # Assume only a sequence of answers was passed
        true_query = None
        true_response = utils.check_response(true_response, result_format='boolean')
    else:
        true_query, true_response = utils.check_query_response(true_response, result_format='list-boolean')

    if not isinstance(pred_response, (sparse.COO, scipy.sparse.spmatrix)) \
            and np.asarray(pred_response).ndim == 1:
        # Assume only a sequence of answers was passed
        pred_triplets, pred_answers = None, embedding_or_pred_answers.astype(int)
    elif isinstance(embedding_or_pred_answers, (np.ndarray, list)) and len(embedding_or_pred_answers) != len(triplets):
        # Assume an embedding was passed
        embedding = check_array(pred_response, ensure_2d=True)
        pred_query, pred_response = datasets.triplet_answers(true_query, embedding, distance='euclidean',
                                                             result_format='list-boolean')
    else:
        # Assume a complete triplet question+answer was passed
        pred_query, pred_response = utils.check_query_response(pred_response, result_format='list-boolean')

    if pred_triplets is not None and np.any(triplets != pred_triplets):
        raise ValueError("Expects identical questions for true and predicted triplets.")
    return metrics.accuracy_score(true_answers, pred_answers)


def query_error(true_answers: utils.Answers, embedding_or_pred_answers: utils.Answers) -> float:
    return 1 - query_accuracy(true_answers, embedding_or_pred_answers)


def _scorer(true_response, query):
    query, pred_response = utils.check_query_response(query, result_format='list-boolean')
    return query_accuracy(true_response, pred_response)


QueryScorer = metrics.make_scorer(_scorer)
