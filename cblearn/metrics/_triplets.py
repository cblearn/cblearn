import numpy as np
from sklearn.utils import check_array
from sklearn import metrics
import sparse
import scipy

from cblearn import utils
from .. import datasets


def query_accuracy(true_response: utils.Response, pred_response: utils.Response) -> float:
    """Fraction of violated triplet constraints.

    For all triplets (i, j, k), count R * (||O(j) - O(i)|| - ||O(k) - O(i)||) > 0
    and divide by the number of triplets.

    Args:
        true_response: Triplet constraints either in array or sparse matrix format
        pred_response: Either object coordinates, shape (n_objects, n_features),
                       or predicted triplet response.
    Returns:
        Number between 0 and 1, indicating the fraction of triplet constraints which are violated.
    """
    if not isinstance(true_response, (sparse.COO, scipy.sparse.spmatrix)) and np.asarray(true_response).ndim == 1:
        # Assume only a sequence of responses was passed
        true_query = None
        true_response = utils.check_response(true_response, result_format='boolean')
    else:
        true_query, true_response = utils.check_query_response(true_response, result_format='list-boolean')

    if not isinstance(pred_response, (sparse.COO, scipy.sparse.spmatrix)) \
            and np.asarray(pred_response).ndim == 1:
        # Assume only a sequence of answers was passed
        pred_query = None
        pred_response = utils.check_response(pred_response, result_format='boolean')
    elif true_query is not None and isinstance(pred_response, (np.ndarray, list)) \
            and len(pred_response) != len(true_query):
        # Assume an embedding was passed
        embedding = check_array(pred_response, ensure_2d=True)
        pred_query, pred_response = datasets.triplet_response(true_query, embedding, distance='euclidean',
                                                              result_format='list-boolean')
    else:
        # Assume a complete triplet query+response was passed
        pred_query, pred_response = utils.check_query_response(pred_response, result_format='list-boolean')

    # sort both triplet lists
    if true_query is not None and pred_query is not None:
        true_ix, pred_ix = np.lexsort(true_query.T), np.lexsort(pred_query.T)
        true_query, true_response = true_query[true_ix], true_response[true_ix]
        pred_query, pred_response = pred_query[pred_ix], pred_response[pred_ix]
        if np.any(true_query != pred_query):
            raise ValueError("Expects identical queries for true and predicted.")
    elif not (true_query is None and pred_query is None):
        raise ValueError("Expects either only responses or query-response pairs for both true and predicted. "
                         "Do not mix these to prevent unexpected behaviour.")

    return metrics.accuracy_score(true_response, pred_response)


def query_error(true_response: utils.Response, pred_response: utils.Response) -> float:
    return 1 - query_accuracy(true_response, pred_response)


def _scorer(true_response, query):
    query, pred_response = utils.check_query_response(query, result_format='list-boolean')
    return query_accuracy(true_response, pred_response)


QueryScorer = metrics.make_scorer(_scorer)
