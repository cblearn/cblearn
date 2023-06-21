from numpy.typing import ArrayLike
from sklearn import metrics


def query_accuracy(true_response: ArrayLike, pred_response: ArrayLike) -> float:
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
    return metrics.accuracy_score(true_response, pred_response)


def query_error(true_response: ArrayLike, pred_response: ArrayLike) -> float:
    return 1 - query_accuracy(true_response, pred_response)


QueryScorer = metrics.make_scorer(query_accuracy)
