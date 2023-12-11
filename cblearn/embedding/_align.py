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


def generalized_procrustes(embeddings: list[ArrayLike], initial_reference=None,
                           threshold: float = 1e-6, max_iter=100):
    """Align multiple embeddings using the generalized procrustes analysis.

    The procrustes alignment is the translation, rotation (including flipping), and scaling
    that minimizes the sum of squared errors between two standardized embeddings (Froebenius norm).

    The embeddings are standardizes by centering on their mean, scaling by the Froebenius norm,
    and rotating to the support vectors of the reference embedding.
    The returned reference is the center of all embeddings in the previous iteration.
    The disparity is the sum of squared residuals between the
    center of aligned embeddings and the reference.

    Args:
        embeddings: List of embeddings to align (first entry is the reference).
        initial_reference: Initial reference embedding (default: first embedding).
        threshold: Stop alignment when the mean disparity is below this threshold.
        max_iter: Maximum number of iterations.
    Returns:
        (reference, aligned_embeddings, disparities)

    >>> A = np.random.rand(10, 2)
    >>> B = A * 0.3 + 0.5  # scale, then translate
    >>> angle_radians = np.radians(30)
    >>> rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
    ...                             [np.sin(angle_radians), np.cos(angle_radians)]])
    >>> C = np.dot(A, rotation_matrix) * 50 + 2  # rotate, scale, translate
    >>> ref, (a, b, c), disp = generalized_procrustes([A, B, C])
    >>> np.allclose(a, b), np.allclose(a, c), np.allclose(b, c), np.allclose(disp, [0, 0])
    (True, True, True, True)
    """
    if initial_reference is None:
        reference = embeddings[0]  # arbitrary choice
    else:
        reference = initial_reference
    reference, __, __ = procrustes_standardize(reference, True)

    embeddings_ = np.empty((len(embeddings), *reference.shape), dtype=np.double)
    for i in range(len(embeddings)):
        embeddings_[i, :, :] = np.asarray(embeddings[i])
        embeddings_[i] -= np.mean(embeddings_[i], 0)
        embeddings_[i] /= np.linalg.norm(embeddings_[i])

    # initial alignment to arbitrary embedding
    for e in embeddings_:
        R, scale = orthogonal_procrustes(reference, e)
        e[:] = (e @ R.T) * scale
    center = np.mean(embeddings_, 0)
    prev_disparity = ((reference - center)**2).sum()
    reference = center

    # iterative alignment to center of embeddings
    for _ in range(max_iter):
        for e in embeddings_:
            R, scale = orthogonal_procrustes(reference, e)
            e[:] = (e @ R.T) * scale
        center = np.mean(embeddings_, 0)
        disparity = ((reference - center)**2).sum()
        if (prev_disparity - disparity) > threshold:
            prev_disparity = disparity
            reference = center
        else:
            break

    return reference, embeddings_, disparity