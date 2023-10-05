from typing import Union

from sklearn.base import BaseEstimator
import numpy as np

from cblearn.embedding._base import QuadrupletEmbeddingMixin


class FORTE(BaseEstimator, QuadrupletEmbeddingMixin):
    """ Fast Ordinal Triplet Embedding (FORTE).

        FORTE [1]_ minimizes a kernel version of the triplet hinge soft objective
        as a smooth relaxation of the triplet error.

        This estimator supports multiple implementations which can be selected by the `backend` parameter.

        The *torch* backend uses the ADAM optimizer and backpropagation [2]_.
        It can executed on CPU, but also CUDA GPUs. We optimize using BFSGS and Strong-Wolfe line search.

        .. Note::
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
        >>> estimator = FORTE(n_components=2)
        >>> embedding = estimator.fit_transform(triplets)
        >>> embedding.shape
        (15, 2)
        >>> estimator.score(triplets) > 0.6
        True


        References
        ----------
        .. [1] Jain, L., Jamieson, K. G., & Nowak, R. (2016). Finite Sample Prediction and
               Recovery Bounds for Ordinal Embedding. Advances in Neural Information Processing Systems, 29.
        .. [2] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, verbose=False, random_state: Union[None, int, np.random.RandomState] = None,
                 learning_rate=0.001, max_iter=2000, batch_size=50_000, device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            verbose: boolean, default=False
                Enable verbose output.
            random_state:
                The seed of the pseudo random number generator used to initialize the optimization.
            learning_rate: step size of the optimization.
            max_iter: Maximum number of optimization iterations.
            batch_size: Batch size of stochastic optimization. Only used with *torch* backend, else ignored.
            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
                Only used with the *torch* backend, else ignored.
        """
        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate,
            kernel=True,
            use_quadruplets=False,
            torch_optimizer='l-bfgs-b',
            torch_kernel_kwargs={'line_search_fn': 'strong_wolfe'},
            backend='torch')

    def _torch_kernel_loss(self, kernel_matrix, triplets, *args):
        triplets = triplets.long()
        diag = kernel_matrix.diag()[:, None]
        dist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
        d_ij = dist[triplets[:, 0], triplets[:, 1]].squeeze()
        d_ik = dist[triplets[:, 0], triplets[:, 2]].squeeze()
        return (1 + (d_ij - d_ik).exp()).log().sum()
