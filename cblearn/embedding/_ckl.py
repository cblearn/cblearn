from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding import _torch_utils


class CKL(BaseEstimator, TripletEmbeddingMixin):
    """ Crowd Kernel Learning (CKL) embedding kernel for triplet data.

        CKL [1]_ searches for an Euclidean representation of objects.
        The model is regularized through the rank of the embedding's kernel matrix.

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
                 backend: str = 'scipy', kernel: bool = False, learning_rate=None, batch_size=50000,
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
        self.n_components = n_components
        self.max_iter = max_iter
        self.mu = mu
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.kernel = kernel
        self.verbose = verbose
        self.random_state = random_state
        self.backend = backend
        self.device = device

    def fit(self, X: utils.Query, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'CKL':
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
            init = random_state.multivariate_normal(
                np.zeros(self.n_components), np.eye(self.n_components), size=n_objects)

        if self.backend == 'torch':
            _torch_utils.assert_torch_is_available()
            if self.kernel:
                result = _torch_utils.torch_minimize_kernel(
                    'adam', _ckl_kernel_loss_torch, init, data=[triplets.astype(int)], args=(self.mu,),
                    device=self.device, max_iter=self.max_iter, batch_size=self.batch_size, lr=self.learning_rate or 100,
                    seed=random_state.randint(1))
            else:
                result = _torch_utils.torch_minimize(
                    'adam', _ckl_x_loss_torch, init, data=(triplets.astype(int),), args=(self.mu,),
                    device=self.device, max_iter=self.max_iter, lr=self.learning_rate or 1,
                    seed=random_state.randint(1))
        elif self.backend == "scipy":
            if self.kernel:
                raise ValueError(f"Kernel objective is not available for backend {self.backend}.")

            result = minimize(_ckl_x_loss, init.ravel(), args=(init.shape, triplets, self.mu), method='L-BFGS-B',
                              jac=True, options=dict(maxiter=self.max_iter, disp=self.verbose))
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Try 'scipy' or 'torch' instead.")

        if self.verbose and not result.success:
            print(f"CKL's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self


def _ckl_x_loss(x, x_shape, triplets, mu, float_min=np.finfo(float).tiny):
    X = x.reshape(x_shape)
    n_objects, n_dim = X.shape
    D = distance.squareform(distance.pdist(X, 'sqeuclidean'))

    I, J, K = tuple(triplets.T)
    nom = mu + D[I, K]
    den = 2 * mu + D[I, K] + D[I, J]
    loss = -(np.log(np.maximum(nom, float_min)) - np.log(np.maximum(den, float_min))).sum()

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

    return loss, loss_grad.ravel()


def _ckl_x_loss_torch(embedding, triplets, mu):
    X = embedding[triplets.long()]
    x_i, x_j, x_k = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    nominator = (x_i - x_k).norm(p=2, dim=1) ** 2 + mu
    denominator = (x_i - x_j).norm(p=2, dim=1) ** 2 + (x_i - x_k).norm(p=2, dim=1) ** 2 + 2 * mu
    return -1 * (nominator.log() - denominator.log()).sum()


def _ckl_kernel_loss_torch(kernel_matrix, triplets, mu):
    triplets = triplets.long()
    diag = kernel_matrix.diag()[:, None]
    dist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
    d_ij = dist[triplets[:, 0], triplets[:, 1]].squeeze()
    d_ik = dist[triplets[:, 0], triplets[:, 2]].squeeze()
    probability = (d_ik + mu).log() - (d_ij + d_ik + 2 * mu).log()
    return -probability.sum()
