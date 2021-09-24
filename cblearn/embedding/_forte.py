from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding import _torch_utils


class FORTE(BaseEstimator, TripletEmbeddingMixin):
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
                 max_iter=2000, batch_size=50_000, device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            verbose: boolean, default=False
                Enable verbose output.
            random_state:
                The seed of the pseudo random number generator used to initialize the optimization.
            max_iter: Maximum number of optimization iterations.
            batch_size: Batch size of stochastic optimization. Only used with *torch* backend, else ignored.
            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
                Only used with the *torch* backend, else ignored.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.batch_size = batch_size

    def fit(self, X: utils.Query, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'FORTE':
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
            init = random_state.multivariate_normal(np.zeros(self.n_components),
                                                    np.eye(self.n_components), size=n_objects)

        _torch_utils.assert_torch_is_available()
        result = _torch_utils.torch_minimize_kernel('l-bfgs-b', _torch_forte_loss, init, data=(triplets.astype(int),),
                                                    device=self.device, max_iter=self.max_iter,
                                                    seed=random_state.randint(1),
                                                    batch_size=self.batch_size, line_search_fn='strong_wolfe')

        if self.verbose and not result.success:
            print(f"FORTE's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self


def _torch_forte_loss(kernel_matrix, triplets):
    triplets = triplets.long()
    diag = kernel_matrix.diag()[:, None]
    dist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
    d_ij = dist[triplets[:, 0], triplets[:, 1]].squeeze()
    d_ik = dist[triplets[:, 0], triplets[:, 2]].squeeze()
    return (1 + (d_ij - d_ik).exp()).log().sum()
