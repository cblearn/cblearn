import numpy as np
import scipy


def procrustes_distance(true_embedding: np.ndarray, pred_embedding: np.ndarray) -> float:
    """ Distance measure between embeddings under optimal transformation.

    The predicted embedding is transformed by Procrustes analysis in terms of
    scaling, rotations, and reflections. The transformation is optimized based on
    the sum of pointwise square-differences between predicted and true embedding.

    Args:
        true_embedding: True object coordinates, (n_objects, n_components)
        pred_embedding: Predicted object coordinates, (n_objects, n_components)

    Returns:
        Sum of pointwise square-differences between the embeddings under optimal transformation.

    Raises:
        ValueError as described in :func:scipy.spatial.procrustes:
    """
    __, __, m_squared = scipy.spatial.procrustes(true_embedding, pred_embedding)
    return m_squared
