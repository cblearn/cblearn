import numpy as np
from sklearn.utils import check_array
from sklearn import metrics

from ..utils import check_triplets, IndexTriplets


def triplet_error(triplets: IndexTriplets, embedding: np.ndarray, responses: np.ndarray = None) -> float:
    """Fraction of violated triplet constraints.

    For all triplets (i, j, k), count R * (||O(j) - O(i)|| - ||O(k) - O(i)||) > 0
    and divide by the number of triplets.

    Args:
        triplets: Triplet constraints either in array or sparse matrix format
        embedding: True object coordinates, (n_objects, n_features)
        responses: Optional, explicit responses to triplet constraints

    Returns:
        Number between 0 and 1, indicating the fraction of triplet constraints
        which are violated.
    """
    print(embedding.shape)
    embedding = check_array(embedding, ensure_2d=True)
    triplets, responses = check_triplets(triplets, responses, format='array', response_type='numeric')

    input_dim = embedding.shape[1]
    triplet_features = embedding[triplets.ravel()].reshape(-1, 3 * input_dim)
    pivot = triplet_features[:, 0:input_dim]
    distances = (np.linalg.norm(pivot - triplet_features[:, input_dim:(2 * input_dim)], axis=1)
                 - np.linalg.norm(pivot - triplet_features[:, (2 * input_dim):], axis=1))
    return np.mean(distances * responses > 0)


TripletScorer = metrics.make_scorer(lambda *args: 1 - metrics.zero_one_loss(*args))
