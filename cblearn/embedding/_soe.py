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

    def __init__(self, n_components=2, margin=0.1, n_init=10, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 restart_optim: int = 10, backend: str = "scipy",
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
            restart_optim:
                Number of restarts at different initial parameters if optimization fails.
                Ignored if an init array is passed at fit method.
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
        self.n_init = n_init
        self.max_iter = max_iter
        self.restart_optim = restart_optim
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
            init: Initial embedding for optimization.
                  Pass a list to run the optimization multiple times and
                  return the best result.
        Returns:
            self.
        """
        queries = utils.check_query_response(X, y, result_format='list-order')
        if not n_objects:
            n_objects = queries.max() + 1
        random_state = check_random_state(self.random_state)
        if init is None:
            inits = (random_state.multivariate_normal(np.zeros(self.n_components),
                     np.eye(self.n_components), size=n_objects) for _ in range(self.n_init))
        else:
            init = np.array(init)
            if init.ndim == 3:
                inits = init
            else:
                inits = [init]

        best_result = None
        for init in inits:
            if self.backend == "torch":
                assert_torch_is_available()
                if queries.shape[1] != 3:
                    raise ValueError(f"Expect triplets of shape (n_triplets, 3), got {queries.shape}.")
                result = torch_minimize('adam', _soe_loss_torch, init, data=(queries.astype(int),), args=(self.margin,),
                                        device=self.device, max_iter=self.max_iter, lr=self.learning_rate,
                                        seed=random_state.randint(1))
            elif self.backend == "scipy":
                if queries.shape[1] == 3:
                    queries = queries[:, [0, 1, 0, 2]]
                elif queries.shape[1] != 4:
                    raise ValueError(f"Expect triplets or quadruplets of shape (n_queries, 3/4), got {queries.shape}.")
                result = minimize(_soe_loss, init.ravel(), args=(init.shape, queries, self.margin), method='L-BFGS-B',
                                  jac=True, options=dict(maxiter=self.max_iter, disp=self.verbose))

            else:
                raise ValueError(f"Unknown backend '{self.backend}'. Try 'scipy' or 'torch' instead.")

            if best_result is None or best_result.fun > result.fun:
                best_result = result
            if self.verbose and not result.success:
                print(f"SOE's optimization failed: {result.message}.\n"
                      f"{'Retry with another initialization...' if init != inits[-1] else ''}")

        self.embedding_ = best_result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = best_result.fun, best_result.nit
        return self


def _soe_loss_torch(embedding, triplets, margin):
    """ Equation (1) of Terada & Luxburg (2014) """
    import torch  # Pytorch is an optional dependency

    X = embedding[triplets.long()]
    anchor, positive, negative = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    triplet_loss = torch.nn.functional.triplet_margin_loss(anchor, positive, negative,
                                                           margin=margin, p=2, reduction='none')
    return torch.sum(triplet_loss**2)


def _soe_loss(x, x_shape, quadruplet, margin):
    """ Loss equation (1) of Terada & Luxburg (2014)
     and Gradient of the loss function.
     """
    # OBJECTIVE #
    X = x.reshape(x_shape)
    X_dist = distance_matrix(X, X)
    ij_dist = X_dist[quadruplet[:, 0], quadruplet[:, 1]]
    kl_dist = X_dist[quadruplet[:, 2], quadruplet[:, 3]]
    differences = ij_dist + margin - kl_dist
    stress = (np.maximum(differences, 0) ** 2)

    # GRADIENT #
    is_diff_positive = differences > 0  # Case 1, 2.1.1
    ij_dist_valid = np.maximum(ij_dist[is_diff_positive, np.newaxis], 0.0000001)
    kl_dist_valid = np.maximum(kl_dist[is_diff_positive, np.newaxis], 0.0000001)
    double_dist = 2 * differences[is_diff_positive, np.newaxis]
    i, j, k, l = quadruplet[is_diff_positive].T

    i_is_k = (i == k)[:, np.newaxis]
    i_is_l = (i == l)[:, np.newaxis]
    j_is_k = (j == k)[:, np.newaxis]
    j_is_l = (j == l)[:, np.newaxis]
    # gradients of distances
    Xij = (X[i] - X[j]) / ij_dist_valid
    Xik = (X[i] - X[k]) / kl_dist_valid  # if i == l
    Xil = (X[i] - X[l]) / kl_dist_valid  # if k == l
    Xjk = (X[j] - X[k]) / kl_dist_valid  # if j == l
    Xjl = (X[j] - X[l]) / kl_dist_valid
    Xkl = (X[k] - X[l]) / kl_dist_valid

    grad = np.zeros_like(X)
    np.add.at(grad, i, double_dist * (Xij - np.where(i_is_k, Xil, np.where(i_is_l, Xik, 0))))
    np.add.at(grad, j, double_dist * (-Xij - np.where(j_is_k, Xjl, np.where(j_is_l, Xjk, 0))))
    np.add.at(grad, k, double_dist * np.where(i_is_k | j_is_k, 0, -Xkl))
    np.add.at(grad, l, double_dist * np.where(i_is_l | j_is_l, 0, Xkl))
    return stress.mean(), grad.ravel() / len(quadruplet)
