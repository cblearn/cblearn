from typing import Union

from sklearn.base import BaseEstimator
import numpy as np

from cblearn.embedding._base import QuadrupletEmbeddingMixin


class GNMDS(BaseEstimator, QuadrupletEmbeddingMixin):
    """ Generalized Non-metric Multidimensional Scaling (GNMDS).

        Embedding estimator for triplet and quadruplet data.

        GNMDS [1]_ minimizes a kernel version of the triplet hinge soft objective
        as a smooth relaxation of the triplet error.

        This estimator supports multiple implementations which can be selected by the `backend` parameter.

        The *torch* backend uses the ADAM optimizer and backpropagation [2]_.
        It can executed on CPU, but also CUDA GPUs.

        .. note::
            The *torch* backend requires the *pytorch* python package (see :ref:`extras_install`).

        Attributes:
            embedding_: Final embedding, shape (n_objects, n_components)
            stress_: Final value of the SOE stress corresponding to the embedding.
            n_iter_: Final number of optimization steps.

        Examples:

        >>> from cblearn import datasets
        >>> np.random.seed(42)
        >>> true_embedding = np.random.rand(15, 2)
        >>> triplets = datasets.make_random_triplets(true_embedding, result_format='list-order', size=1000)
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
        >>> estimator = GNMDS(n_components=2)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> round(estimator.score(triplets), 1) > 0.6
        True
        >>> estimator = GNMDS(n_components=2, backend='torch')
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> round(estimator.score(triplets), 1) > 0.6
        True

        References
        ----------
        .. [1] Agarwal, S., Wills, J., Cayton, L., Lanckriet, G., Kriegman, D., & Belongie, S. (2007).
               Generalized non-metric multidimensional scaling. Artificial Intelligence and Statistics, 11–18.
        .. [2] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, lambd=0.0, margin=1, verbose=False, kernel=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=2000, backend: str = 'scipy',
                 learning_rate=0.0002, batch_size=50_000, device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            lambd: Regularization parameter. The strength of the rank regularization is proportional to lambda.
            margin: Margin parameter, scaling the embedding.
            verbose: Enable verbose output.
            random_state: The seed of the pseudo random number generator used to initialize the optimization.
            max_iter: Maximum number of optimization iterations.
            backend: The optimization backend for fitting. {"scipy", "torch"}
            learning_rate: Learning rate of the gradient-based optimizer.
                           Only used with *torch* backend, else ignored.
            batch_size: Batch size of stochastic optimization. Only used with *torch* backend, else ignored.
            device:
                 The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
                Only used with the *torch* backend, else ignored.
        """
        self.margin = margin
        self.lambd = lambd
        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate,
            backend=backend,
            kernel=kernel,
            use_quadruplets=True)

    def _scipy_loss(self, embedding, quads, dist_pos, dist_neg, *args):
        """ Calculate the loss and gradient for scipy.optimize.minimize.

        The GNMDS loss is similar to the SOE loss.
        Main differrences are (a) using squared distances
                              (b) no squared distance difference
                              (c) add regularization lambda
        """
        # OBJECTIVE #
        differences = (dist_pos**2 - dist_neg**2 + self.margin)
        loss = np.maximum(differences, 0)
        loss = loss.mean() + self.lambd * (embedding**2).mean()

        # GRADIENT #
        is_diff_positive = differences > 0  # Case 1, 2.1.1
        i, j, k, l = quads[is_diff_positive].T

        i_is_k = (i == k)[:, np.newaxis]
        i_is_l = (i == l)[:, np.newaxis]
        j_is_k = (j == k)[:, np.newaxis]
        j_is_l = (j == l)[:, np.newaxis]
        # gradients of distances
        Xij = 2 * (embedding[i] - embedding[j])
        Xik = 2 * (embedding[i] - embedding[k])  # if i == l
        Xil = 2 * (embedding[i] - embedding[l])  # if k == l
        Xjk = 2 * (embedding[j] - embedding[k])  # if j == l
        Xjl = 2 * (embedding[j] - embedding[l])
        Xkl = 2 * (embedding[k] - embedding[l])

        grad = np.zeros_like(embedding)
        np.add.at(grad, i, (Xij - np.where(i_is_k, Xil, np.where(i_is_l, Xik, 0))))
        np.add.at(grad, j, (-Xij - np.where(j_is_k, Xjl, np.where(j_is_l, Xjk, 0))))
        np.add.at(grad, k, np.where(i_is_k | j_is_k, 0, -Xkl))
        np.add.at(grad, l, np.where(i_is_l | j_is_l, 0, Xkl))

        grad = grad / len(quads) + self.lambd * 2 * embedding
        return loss.mean(), grad.ravel()

    def _torch_kernel_loss(self, kernel_matrix, quads, *args):
        quads = quads.long()
        diag = kernel_matrix.diag()[:, None]
        dist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
        d_ij = dist[quads[:, 0], quads[:, 1]].squeeze()
        d_kl = dist[quads[:, 2], quads[:, 3]].squeeze()
        return (d_ij - d_kl).clamp(min=0).mean()\
            + self.lambd * kernel_matrix.trace().mean()

    def _torch_loss(self, embedding, triplets, dist_pos, dist_neg, *args):
        loss = (dist_pos**2 + 1 - dist_neg**2).clamp(min=0)
        return loss.mean() + self.lambd * (embedding**2).mean()
