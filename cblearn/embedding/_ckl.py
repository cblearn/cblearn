from typing import Union

from sklearn.base import BaseEstimator
import numpy as np

from cblearn.embedding._base import QuadrupletEmbeddingMixin


class CKL(BaseEstimator, QuadrupletEmbeddingMixin):
    """ Crowd Kernel Learning (CKL) embedding kernel for triplet data.

        CKL [1]_ searches for an Euclidean representation of objects.
        The model is regularized through the rank of the embedding's kernel matrix.


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
        >>> estimator = CKL(n_components=2)
        >>> embedding = estimator.fit_transform(triplets)
        >>> embedding.shape
        (15, 2)
        >>> round(estimator.score(triplets), 1) > 0.6
        True
        >>> estimator = CKL(n_components=2, backend='torch', kernel=True)
        >>> embedding = estimator.fit_transform(triplets)
        >>> embedding.shape
        (15, 2)


        References
        ----------
        .. [1] Tamuz, O., & Liu, mu., & Belognie, S., & Shamir, O., & Kalai, A.T. (2011).
               Adaptively Learning the Crowd Kernel. International Conference on Machine Learning.
        .. [2] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, mu=0.0, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=2000,
                 backend: str = 'scipy', kernel: bool = False, learning_rate=0.002, batch_size=50000,
                 device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components: The dimension of the embedding.
            mu: Regularization parameter >= 0. Increased mu serves as increasing a margin constraint.
            verbose: Enable verbose output.
            random_state: The seed of the pseudo random number generator used to initialize the optimization.
            max_iter: Maximum number of optimization iterations.
            backend: The optimization backend for fitting. {"torch"}
            kernel: Whether to optimize the kernel or the embedding (default).
            learning_rate: Learning rate of the gradient-based optimizer.
                           If None, then 100 is used, or 1 if kernel=True.
                           Only used with *torch* backend, else ignored.
            batch_size: Batch size of stochastic optimization. Only used with the *torch* backend, else ignored.
            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
                Only used with the *torch* backend, else ignored.
        """
        self.mu = mu
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
            use_quadruplets=False)

    def _scipy_loss(self, X, triplets, D_pos, D_neg, *args):
        float_min = np.finfo(float).tiny
        n_objects, n_dim = X.shape

        I, J, K = tuple(triplets.T)
        nom = self.mu + D_neg**2
        den = 2 * self.mu + D_pos**2 + D_neg*2
        loss = -(np.log(np.maximum(nom, float_min)) - np.log(np.maximum(den, float_min)))

        loss_grad = np.empty_like(X)
        for dim in range(n_dim):
            triplet_grads = [
                2 / nom * (X[I, dim] - X[K, dim]) - 2 / den * ((X[I, dim] - X[J, dim]) + (X[I, dim] - X[K, dim])),
                2 / den * (X[I, dim] - X[J, dim]),
                -2 / nom * (X[I, dim] - X[K, dim]) + 2 / den * (X[I, dim] - X[K, dim]),
                ]
            loss_grad[:, dim] = -np.bincount(triplets[:, 0], triplet_grads[0], n_objects)
            loss_grad[:, dim] -= np.bincount(triplets[:, 1], triplet_grads[1], n_objects)
            loss_grad[:, dim] -= np.bincount(triplets[:, 2], triplet_grads[2], n_objects)

        return loss.mean(), loss_grad / len(triplets)

    def _torch_loss(self, embedding, triplets, dist_pos, dist_neg, *args):
        nom = dist_neg**2 + self.mu
        denom = nom + dist_pos**2 + self.mu
        return -1 * (nom.log() - denom.log()).mean()

    def _torch_kernel_loss(self, kernel_matrix, quads, *args):
        quads = quads.long()
        diag = kernel_matrix.diag()[:, None]
        sqdist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
        d_ij = sqdist[quads[:, 0], quads[:, 1]].squeeze()
        d_ik = sqdist[quads[:, 0], quads[:, 2]].squeeze()
        probs = (d_ik + self.mu).log() - (d_ij + d_ik + 2 * self.mu).log()
        return -probs.mean()
