from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding import _torch_utils


class GNMDS(BaseEstimator, TripletEmbeddingMixin):
    """ Generalized Non-metric Multidimensional Scaling (GNMDS).

        Embedding estimator for triplet and quadruplet data (currently only triplet data is implemented).

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
        >>> estimator = GNMDS(n_components=2, backend='torch', kernel=True)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> round(estimator.score(triplets), 1) > 0.6
        True

        References
        ----------
        .. [1] Agarwal, S., Wills, J., Cayton, L., Lanckriet, G., Kriegman, D., & Belongie, S. (2007).
               Generalized non-metric multidimensional scaling. Artificial Intelligence and Statistics, 11–18.
        .. [2] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, lambd=0.0, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=2000, backend: str = 'scipy',
                 kernel: bool = False, learning_rate=10, batch_size=50_000, device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            lambd: Regularization parameter. The strength of the rank regularization is proportional to lambda.
            verbose: Enable verbose output.
            random_state: The seed of the pseudo random number generator used to initialize the optimization.
            max_iter: Maximum number of optimization iterations.
            backend: The optimization backend for fitting. {"scipy", "torch"}
            kernel: Whether to optimize in kernel or embedding (default) space.
            learning_rate: Learning rate of the gradient-based optimizer.
                           Only used with *torch* backend, else ignored.
            batch_size: Batch size of stochastic optimization. Only used with *torch* backend, else ignored.
            device:
                 The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
                Only used with the *torch* backend, else ignored.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.kernel = kernel
        self.device = device
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.backend = backend

    def fit(self, X: utils.Query, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'GNMDS':
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

        if self.backend == 'torch':
            _torch_utils.assert_torch_is_available()
            if self.kernel:
                result = _torch_utils.torch_minimize_kernel(
                    'adam', _gnmds_kernel_loss_torch, init, data=(triplets.astype(int),), args=(self.lambd,),
                    device=self.device, max_iter=self.max_iter, batch_size=self.batch_size, seed=random_state.randint(1),
                    lr=self.learning_rate)
            else:
                result = _torch_utils.torch_minimize('adam', _gnmds_x_loss_torch, init, data=(triplets.astype(int),),
                                                     args=(self.lambd,), device=self.device, max_iter=self.max_iter,
                                                     seed=random_state.randint(1), lr=self.learning_rate)
        elif self.backend == "scipy":
            if self.kernel:
                raise ValueError(f"Kernel objective is not available for backend {self.backend}.")

            result = minimize(_gnmds_x_grad, init.ravel(), args=(init.shape, triplets, self.lambd), method='L-BFGS-B',
                              jac=True, options=dict(maxiter=self.max_iter, disp=self.verbose))
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Try 'scipy' or 'torch' instead.")

        if self.verbose and not result.success:
            print(f"GNMDS's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self


def _gnmds_x_grad(x, x_shape, triplets, lambd):
    X = x.reshape(x_shape)  # scipy minimize expects a flat x.
    n_objects, n_dim = X.shape
    D = distance.squareform(distance.pdist(X, 'sqeuclidean'))

    I, J, K = tuple(triplets.T)
    slack = np.maximum(D[I, J] + 1 - D[I, K], 0)
    loss = slack.sum() + lambd * (X**2).sum()

    loss_grad = np.empty_like(X)
    triplets = triplets[slack > 0]
    I, J, K = tuple(triplets.T)
    for dim in range(X.shape[1]):
        loss_grad[:, dim] = np.bincount(triplets[:, 0], 2 * (X[I, dim] - X[J, dim])
                                        - 2 * (X[I, dim] - X[K, dim]), n_objects)
        loss_grad[:, dim] += np.bincount(triplets[:, 1], -2 * (X[I, dim] - X[J, dim]), n_objects)
        loss_grad[:, dim] += np.bincount(triplets[:, 2], 2 * (X[I, dim] - X[K, dim]), n_objects)
    loss_grad = loss_grad + lambd * 2 * X

    return loss, loss_grad.ravel()


def _gnmds_kernel_loss_torch(kernel_matrix, triplets, lambd):
    triplets = triplets.long()
    diag = kernel_matrix.diag()[:, None]
    dist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
    d_ij = dist[triplets[:, 0], triplets[:, 1]].squeeze()
    d_ik = dist[triplets[:, 0], triplets[:, 2]].squeeze()
    return (d_ij - d_ik).clamp(min=0).sum() + lambd * kernel_matrix.trace()


def _gnmds_x_loss_torch(embedding, triplets, lambd, p=2.):
    import torch  # Pytorch is an optional dependency

    X = embedding[triplets.long()]
    I, J, K = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    sqnorm_near = torch.linalg.vector_norm(I - J, ord=p, dim=1)**2
    sqnorm_far = torch.linalg.vector_norm(I - K, ord=p, dim=1)**2
    loss = (sqnorm_near + 1 - sqnorm_far).clamp(min=0).sum()

    return loss + lambd * (X**2).sum()
