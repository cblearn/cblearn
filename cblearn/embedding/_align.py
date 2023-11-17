import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import orthogonal_procrustes


def procrustes_standardize(reference, standardize_rotation):
    # standardize the reference embedding: translation, scale, rotation
    reference = np.array(reference, dtype=np.double, copy=True)
    reference_center = np.mean(reference, 0)
    reference -= reference_center
    reference_scale = np.linalg.norm(reference)
    reference /= reference_scale
    if standardize_rotation:
        U, S, _ = np.linalg.svd(reference, full_matrices=False)
        # flip sign based on absolute value for deterministic results
        max_abs = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs, range(U.shape[1])])
        reference = U * S * signs
    return reference, reference_center, reference_scale


def procrustes_alignment(embeddings: list[ArrayLike], return_disparity=False,
                         keep_reference=False, standardize_rotation=True):
    """Align multiple embeddings using procrustes alignment.

    The procrustes alignment is the translation, rotation (including flipping), and scaling
    that minimizes the sum of squared errors between two standardized embeddings (Froebenius norm).
    The alignment is pairwise calculated between the first embedding (the reference) and the others.

    The embeddings are standardizes by centering on their mean, scaling by the Froebenius norm,
    and rotating to the support vectors of the reference embedding.

    Args:
        embeddings: List of embeddings to align (first entry is the reference).
        keep_reference: Do not standardize the reference before alignment.
        standardize_rotation: Standardize rotation of the reference embedding,
                              used only if keep_reference is False.
        return_disparity: Return list of disparity between reference and aligned embeddings.
    Returns:
        Aligned embeddings.
        Tuple of aligned embeddings and disparities, if return_disparity is True.

    >>> A = np.random.rand(10, 2)
    >>> B = A * 0.3 + 0.5  # scale, then translate
    >>> angle_radians = np.radians(30)
    >>> rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
    ...                             [np.sin(angle_radians), np.cos(angle_radians)]])
    >>> C = np.dot(A, rotation_matrix) * 50 + 2  # rotate, scale, translate
    >>> (a, b, c), disp = procrustes_alignment([A, B, C], return_disparity=True)
    >>> np.allclose(a, b), np.allclose(a, c), np.allclose(b, c), np.allclose(disp, [0, 0])
    (True, True, True, True)
    >>> (a, b, c), disp = procrustes_alignment([A, B, C], return_disparity=True, keep_reference=True)
    >>> np.allclose(A, a), np.allclose(a, b), np.allclose(a, c), np.allclose(b, c), np.allclose(disp, [0, 0])
    (True, True, True, True, True)
    """
    reference = np.array(embeddings[0], dtype=np.double, copy=True)

    others = []
    for e in embeddings[1:]:
        others.append(np.array(e, dtype=np.double, copy=True))
    reference, reference_center, reference_scale = procrustes_standardize(
        reference, not keep_reference and standardize_rotation)

    # align others translation, scale, rotation
    for other in others:  # inplace operations
        other -= np.mean(other, 0)
        other /= np.linalg.norm(other)
        R, scale = orthogonal_procrustes(reference, other)
        other[:] = (other @ R.T) * scale
        if keep_reference:
            other *= reference_scale
            other += reference_center

    if keep_reference:
        reference *= reference_scale
        reference += reference_center

    if return_disparity:
        disparities = np.array([((reference - other)**2).sum() for other in others])
        return np.array([reference] + others), disparities
    else:
        return np.array([reference] + others)