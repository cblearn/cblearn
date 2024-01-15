from typing import Union

from sklearn.base import BaseEstimator
import numpy as np

from cblearn.embedding._base import QuadrupletEmbeddingMixin


class SOE(BaseEstimator, QuadrupletEmbeddingMixin):
    """ Soft Ordinal Embedding (SOE).

        SOE [1]_ is minimizing the soft objective as a smooth relaxation of the triplet error.

        This estimator supports multiple implementations which can be selected by the `backend` parameter.
        Both triplet and quadruplet data is supported.

        This class restarts the optimizition from multiple random initializations to increase the probability of
        good results in low-dimensional embeddings. This behaviour multiplies computation time when fitting
        the embeding but can be disabled by `SOE(n_components, n_init=1)`.

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
        >>> seed = np.random.RandomState(42)
        >>> true_embedding = seed.rand(15, 2)
        >>> triplets = datasets.make_random_triplets(true_embedding, result_format='list-order',
        ...                                          size=500, random_state=seed)
        >>> triplets.shape, np.unique(triplets).shape
        ((500, 3), (15,))
        >>> estimator = SOE(n_components=2, random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> round(estimator.score(triplets), 1)
        1.0

        The following is running on the CUDA GPU, if available (but requires pytorch installed).

        >>> estimator = SOE(n_components=2, backend="torch", random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> round(estimator.score(triplets), 1) > 0.6
        True

        References
        ----------
        .. [1] Terada, Y., & Luxburg, U. (2014). Local ordinal embedding.
               International Conference on Machine Learning, 847–855.
        .. [2] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, margin=0.1, n_init=10, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy",
                 learning_rate=1, batch_size=50_000,  device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            margin: Scale parameter which only takes strictly positive value.
                Defines the intended minimal difference of distances in the embedding space between
                for any triplet.
            n_init: repeat the optimization procedure n_init times.
            verbose: boolean, default=False
                Enable verbose output.
            random_state:
             The seed of the pseudo random number generator used to initialize the optimization.
            max_iter:
                Maximum number of optimization iterations.
            backend: The backend used to optimize the objective. {"scipy", "torch"}
            learning_rate: Learning rate of the gradient-based optimizer.
                           If None, then 100 is used, or 1 if kernel=True.
                           Only used with *torch* backend, else ignored.
            batch_size: Batch size of stochastic optimization. Only used with the *torch* backend, else ignored.
            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
                Only used with the *torch* backend, else ignored.
        """
        self.margin = margin
        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate,
            backend=backend,
            n_init=n_init,
            use_quadruplets=True)

    def _scipy_loss(self, embedding, quads, dist_pos, dist_neg, sample_weight, *args):
        """ Loss equation (1) of Terada & Luxburg (2014)
         and Gradient of the loss function.
         """
        # OBJECTIVE #
        differences = sample_weight * (dist_pos - dist_neg + self.margin)
        stress = (np.maximum(differences, 0)**2)

        # GRADIENT #
        is_diff_positive = differences > 0  # Case 1, 2.1.1
        ij_dist_valid = np.maximum(dist_pos[is_diff_positive, np.newaxis], 0.0000001)
        kl_dist_valid = np.maximum(dist_neg[is_diff_positive, np.newaxis], 0.0000001)
        double_dist = 2 * differences[is_diff_positive, np.newaxis]
        i, j, k, l = quads[is_diff_positive].T

        i_is_k = (i == k)[:, np.newaxis]
        i_is_l = (i == l)[:, np.newaxis]
        j_is_k = (j == k)[:, np.newaxis]
        j_is_l = (j == l)[:, np.newaxis]
        # gradients of distances
        Xij = (embedding[i] - embedding[j]) / ij_dist_valid
        Xik = (embedding[i] - embedding[k]) / kl_dist_valid  # if i == l
        Xil = (embedding[i] - embedding[l]) / kl_dist_valid  # if k == l
        Xjk = (embedding[j] - embedding[k]) / kl_dist_valid  # if j == l
        Xjl = (embedding[j] - embedding[l]) / kl_dist_valid
        Xkl = (embedding[k] - embedding[l]) / kl_dist_valid

        grad = np.zeros_like(embedding)
        np.add.at(grad, i, double_dist * (Xij - np.where(i_is_k, Xil, np.where(i_is_l, Xik, 0))))
        np.add.at(grad, j, double_dist * (-Xij - np.where(j_is_k, Xjl, np.where(j_is_l, Xjk, 0))))
        np.add.at(grad, k, double_dist * np.where(i_is_k | j_is_k, 0, -Xkl))
        np.add.at(grad, l, double_dist * np.where(i_is_l | j_is_l, 0, Xkl))

        return stress.mean(), grad.ravel() / len(quads)

    def _torch_loss(self, embedding, quads, dist_pos, dist_neg, *args):
        """ Equation (1) of Terada & Luxburg (2014) """
        import torch  # Pytorch is an optional dependency
        return torch.mean(torch.clamp_min(self.margin + dist_pos - dist_neg, 0)**2)
