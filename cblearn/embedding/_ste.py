from scipy.spatial import distance

from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np
from scipy.optimize import minimize

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding._torch_utils import assert_torch_is_available, torch_minimize


class STE(BaseEstimator, TripletEmbeddingMixin):
    """ Stochastic Triplet Embedding algorithm (STE / t-STE).

        STE [1]_ maximizes the probability, that the triplets are satisfied.
        The variant t-STE is using the heavy tailed Student-t-kernel instead of a Gaussian kernel.

        This estimator supports multiple implementations which can be selected by the `backend` parameter.
        The *scipy* backend uses the L-BSGS-B optimizer.
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
        >>> estimator = STE(n_components=2, random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> estimator.score(triplets) > 0.8
        True

        The following is running on the CUDA GPU, if available (but requires pytorch installed).

        >>> estimator = STE(n_components=2, backend="torch", random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> estimator.score(triplets) > 0.8
        True

        References
        ----------
        .. [1] van der Maaten, L., & Weinberger, K. (2012). Stochastic triplet embedding.
               2012 IEEE International Workshop on Machine Learning for Signal Processing, 1–6.
        .. [2] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, heavy_tailed=False, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy", learning_rate=1, batch_size=50_000,  device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            heavy_tailed:
                If false, STE is using the Gaussian kernel,
                If true, t-STE is using the heavy-tailed student-t kernel.
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
        self.heavy_tailed = heavy_tailed
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.backend = backend
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

    def fit(self, X: utils.Query, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'STE':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            init: Initial embedding for optimization
            n_objects: Number of objects in the embedding.
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
            result = torch_minimize('adam', _ste_x_torch, init, data=(triplets.astype(int),), args=(self.heavy_tailed,),
                                    device=self.device, max_iter=self.max_iter, lr=self.learning_rate,
                                    seed=random_state.randint(1))
        elif self.backend == "scipy":
            result = minimize(_ste_x_grad, init.ravel(), args=(init.shape, triplets, self.heavy_tailed),
                              method='L-BFGS-B', jac=True, options=dict(maxiter=self.max_iter, disp=self.verbose))
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Try 'scipy' or 'torch' instead.")

        if self.verbose and not result.success:
            print(f"STE's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self


def _ste_x_torch(embedding, triplets, heavy_tailed, p=2.):
    import torch  # Pytorch is an optional dependency

    X = embedding[triplets.long()]
    dof = max(embedding.shape[1] - 1, 1)
    I, J, K = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    dist_1 = torch.linalg.vector_norm(I - J, ord=p, dim=1)**2
    dist_2 = torch.linalg.vector_norm(I - K, ord=p, dim=1)**2
    if heavy_tailed:
        t_dist_1 = (1 + dist_1 / 2.)**(-(dof + 1) / 2)
        loss = t_dist_1 / (t_dist_1 + (1 + dist_2 / 2.)**(-(dof + 1) / 2) + 1e-16)
    else:
        loss = (-dist_1).exp() / ((-dist_1).exp() + (-dist_2).exp() + 1e-16)

    return -loss.log().sum()


def _ste_x_grad(x, x_shape, triplets, heavy_tailed):
    X = x.reshape(x_shape)  # scipy minimize expects a flat x.
    n_objects, n_dim = X.shape
    dof = max(n_dim - 1, 1)
    dist = distance.squareform(distance.pdist(X, 'sqeuclidean'))
    if heavy_tailed:
        base_kernel = 1 + dist / dof
        kernel = base_kernel**(-(dof + 1) / 2)
    else:
        kernel = np.exp(-dist)

    I, J, K = tuple(triplets.T)
    P = kernel[I, J] / (kernel[I, J] + kernel[I, K])
    loss = -np.log(np.maximum(P, np.finfo(float).tiny)).sum()

    if heavy_tailed:
        base_inv = (1 / base_kernel)[..., np.newaxis]
        grad_triplets = - (dof + 1) / dof * np.array([
            base_inv[I, J] * (X[I] - X[J]) - base_inv[I, K] * (X[I] - X[K]),
            - base_inv[I, J] * (X[I] - X[J]),
            base_inv[I, K] * (X[I] - X[K])])
    else:
        grad_triplets = - 2 * np.array([
            (X[I] - X[J]) - (X[I] - X[K]),
            - (X[I] - X[J]),
            (X[I] - X[K])])

    grad_triplets *= (P * (1 - P))[np.newaxis, :, np.newaxis]

    loss_grad = np.empty_like(X)
    for dim in range(X.shape[1]):
        loss_grad[:, dim] = np.bincount(triplets[:, 0], grad_triplets[0, :, dim], n_objects)
        loss_grad[:, dim] += np.bincount(triplets[:, 1], grad_triplets[1, :, dim], n_objects)
        loss_grad[:, dim] += np.bincount(triplets[:, 2], grad_triplets[2, :, dim], n_objects)
    loss_grad = -loss_grad

    return loss, loss_grad.ravel()


class TSTE(STE):
    """ t-Distributed Stochastic Triplet Embedding (t-STE)

    Variant of :class:`STE`, that assumes t-student distributed distances
    which leads to better optimization properties."""
    def __init__(self, n_components=2, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy", learning_rate=1, batch_size=50_000,  device: str = "auto"):
        heavy_tailed = True
        return super().__init__(n_components, heavy_tailed, verbose, random_state, max_iter, backend,
                                learning_rate, batch_size, device)
