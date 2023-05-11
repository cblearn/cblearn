import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import orthogonal_procrustes
from sklearn.utils import check_array


def procrustes_alignment(embeddings: list[ArrayLike], return_disparity=False,
                         standardize_reference=True, standardize_rotation=True):
    """Align multiple embeddings using procrustes alignment.

    Args:
        embeddings: List of embeddings to align (first entry is the reference).
        standardize_reference: Standardize size and translation of the reference embedding.
        standardize_rotation: Standardize rotation of the reference embedding,
                              used only if standardize_reference is True.
        return_disparity: Return list of disparity between reference and aligned embeddings.
    Returns:
        Aligned embeddings.
        Tuple of aligned embeddings and disparities, if return_disparity is True.

    >>> A = np.random.rand(10, 10)
    >>> B = (A + 0.5) * 0.3
    >>> C = np.ro(A - 0.3) * 0.2
    >>> (A, B, C), disp = procrustes_alignment([A, B, C], return_disparity=True)
    >>> np.allclose(A, B), np.allclose(A, C), np.allclose(B, C), np.allclose(disp, [0, 0])
    (True, True, True, True)
    """
    n_entries = max(0, *map(len, embeddings))
    emb0 = check_array(embeddings[0])
    reference = np.zeros((n_entries, emb0.shape[1]))
    reference[:len(emb0), :] = emb0

    others = []
    for e in embeddings[1:]:
        others.append(np.zeros_like(reference))
        others[-1][:len(e), :] = check_array(e)

    # standardize the reference embedding: translation, scale, rotation
    if standardize_reference:
        reference -= reference.mean()
        reference /= np.linalg.norm(reference)
        if standardize_rotation:
            U, S, _ = np.linalg.svd(reference, full_matrices=False)
            # flip sign based on absolute value for deterministic results
            max_abs = np.argmax(np.abs(U), axis=0)
            signs = np.sign(U[max_abs, range(U.shape[1])])
            reference = U * S * signs

    # align others translation, scale, rotation
    for other in others:  # inplace operations
        other -= other.mean()
        other /= np.linalg.norm(other)
        R, scale = orthogonal_procrustes(reference, other)
        other[:] = scale * (other @ R.T)

    if return_disparity:
        disparities = np.array([((reference - other)**2).sum() for other in others])
        return np.array([reference] + others), disparities
    else:
        return np.array([reference] + others)