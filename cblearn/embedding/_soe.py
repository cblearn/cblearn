from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding._torch_utils import assert_torch_is_available, torch_minimize


class SOE(BaseEstimator, TripletEmbeddingMixin):
    """ Soft Ordinal Embedding (SOE).

        SOE [1]_ is minimizing the soft objective as a smooth relaxation of the triplet error.

        This estimator supports multiple implementations which can be selected by the `backend` parameter.
        The majorizing backend for SOE is described in the paper original paper.

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
        ...                                          size=1000, random_state=seed)
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
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

    def __init__(self, n_components=2, margin=1, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy", learning_rate=1, batch_size=50_000,  device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            margin: Scale parameter which only takes strictly positive value.
                Defines the intended minimal difference of distances in the embedding space between
                for any triplet.
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
        self.n_components = n_components
        self.margin = margin
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.backend = backend
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

    def fit(self, X: utils.Query, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'SOE':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            init: Initial embedding for optimization
        Returns:
            self.
        """
        triplets = utils.check_query_response(X, y, result_format='list-order')
        if not n_objects:
            n_objects = triplets.max() + 1
        random_state = check_random_state(self.random_state)
        if init is None:
            init = random_state.multivariate_normal(np.zeros(self.n_components), np.eye(self.n_components),
                                                    size=n_objects)

        if self.backend == "torch":
            assert_torch_is_available()
            result = torch_minimize('adam', _soe_loss_torch, init, data=(triplets.astype(int),), args=(self.margin,),
                                    device=self.device, max_iter=self.max_iter, lr=self.learning_rate,
                                    seed=random_state.randint(1))
        elif self.backend == "scipy":
            result = minimize(_soe_loss, init.ravel(), args=(init.shape, triplets, self.margin), method='L-BFGS-B',
                              jac=True, options=dict(maxiter=self.max_iter, disp=self.verbose))
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Try 'scipy' or 'torch' instead.")

        if self.verbose and not result.success:
            print(f"SOE's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self


def _soe_loss_torch(embedding, triplets, margin):
    """ Equation (1) of Terada & Luxburg (2014) """
    import torch  # Pytorch is an optional dependency

    X = embedding[triplets.long()]
    anchor, positive, negative = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    triplet_loss = torch.nn.functional.triplet_margin_loss(anchor, positive, negative,
                                                           margin=margin, p=2, reduction='none')
    return torch.sum(triplet_loss**2)


def _soe_loss(x, x_shape, triplets, margin):
    """ Loss equation (1) of Terada & Luxburg (2014)
     Gradient of majorizating function (2) of Terada & Luxburg (2014)
        as described in the supplementary material 2.1.
     """
    X = x.reshape(x_shape)
    X_dist = distance_matrix(X, X)
    ij_dist = X_dist[triplets[:, 0], triplets[:, 1]]
    kl_dist = X_dist[triplets[:, 0], triplets[:, 2]]
    ij_dist, kl_dist = np.maximum(ij_dist, 0.0000001), np.maximum(kl_dist, 0.0000001)
    differences = ij_dist + margin - kl_dist
    stress = (np.maximum(differences, 0) ** 2).sum()

    is_diff_positive = differences > 0  # Case 1, 2.1.1
    ij_dist_valid, kl_dist_valid = ij_dist[is_diff_positive, np.newaxis], kl_dist[is_diff_positive, np.newaxis]
    double_dist = 2 * differences[is_diff_positive, np.newaxis]
    i, j, k = triplets[is_diff_positive, 0], triplets[is_diff_positive, 1], triplets[is_diff_positive, 2]

    i_is_l = (i == k)[:, np.newaxis]
    Xij = (X[i] - X[j]) / ij_dist_valid
    Xil = (X[i] - X[k]) / kl_dist_valid

    grad = np.zeros_like(X)
    np.add.at(grad, i, double_dist * (Xij - Xil + np.where(i_is_l, - Xil, 0)))
    np.add.at(grad, j, double_dist * -Xij)
    np.add.at(grad, k, double_dist * Xil)

    return stress, grad.ravel()
