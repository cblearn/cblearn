from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np
import scipy
from scipy.spatial import distance_matrix

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding import _torch_utils


class GNMDS(BaseEstimator, TripletEmbeddingMixin):
    """ Generalized Non-metric Multidimensional Scaling (GNMDS).

        GNMDS [1]_ minimizes a kernel version of the triplet hinge soft objective as a smooth relaxation of the triplet error.

        This estimator supports multiple implementations which can be selected by the `algorithm` parameter.
        The majorizing algorithm for SOE is described in the paper original paper [1]_.

        An alternative implementation is using backpropagation, like descibed in [2]_.
        This one can run not only on CPU, but also GPU with CUDA. For this, it depends
        on the pytorch package (see :ref:`extras_install`).

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
        >>> estimator = GNMDS(n_components=2, algorithm='K')
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> round(estimator.score(triplets), 1)
        0.7

        The following is running on the CUDA GPU, if available (but requires pytorch installed).

        >>> estimator = GNMDS(n_components=2, algorithm="X")
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> round(estimator.score(triplets), 1)
        1.0

        References
        ----------
        .. [1] Terada, Y., & Luxburg, U. (2014). Local ordinal embedding.
               International Conference on Machine Learning, 847–855.
        .. [2] Vankadara, L. et al. (2019) Insights into Ordinal Embedding Algorithms: A Systematic Evaluation
               Arxiv Preprint, https://arxiv.org/abs/1912.01666
        """

    def __init__(self, n_components=2, max_iter=2000, lambd=0.0, learning_rate=10, batch_size=1000000, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None,
                 algorithm: str = "X", device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            lambd:
                Regularization parameter.
            max_iter:
                Maximum number of optimization iterations.
            verbose: boolean, default=False
                Enable verbose output.
            random_state:
                The seed of the pseudo random number generator used to initialize the optimization.
            algorithm:
                The algorithm used to optimize the soft objective. {"majorizing", "backprop"}
            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
                This parameter is only used if "backprop" algorithm is used.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.algorithm = algorithm
        self.device = device
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, X: utils.Questions, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'GNMDS':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            init: Initial embedding for optimization
        Returns:
            self.
        """
        triplets = utils.check_triplet_answers(X, y, result_format='list-order')
        if not n_objects:
            n_objects = len(np.unique(triplets))
        if init is None:
            random_state = check_random_state(self.random_state)
            init = random_state.multivariate_normal(np.zeros(self.n_components), np.eye(self.n_components),
                                                    size=n_objects)

        if self.algorithm == "K":
            """
            The kernel version of the algorithm.
            """
            _torch_utils.assert_torch_is_available()
            result = _torch_utils.torch_minimize_kernel('adam', _gnmds_kernel_loss_torch, init, data=(triplets.astype(int),),
                                                        args=(self.lambd,), device=self.device,
                                                     max_iter=self.max_iter, batch_size=self.batch_size, lr=self.learning_rate)
        elif self.algorithm == "X":
            """
                The embedding (X) version of the algorithm.
            """
            _torch_utils.assert_torch_is_available()
            result = _torch_utils.torch_minimize('adam', _gnmds_x_loss_torch, init, data=(triplets.astype(int),),args=(self.lambd,),
                                          device=self.device, max_iter=self.max_iter, lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown GNMDS algorithm '{self.algorithm}'. Try 'K' or 'X' instead.")

        if self.verbose and not result.success:
            print(f"GNMDS's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self



def _gnmds_kernel_loss_torch(kernel_matrix, triplets, lambd):
    diag = kernel_matrix.diag()[:, None]
    dist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
    d_ij = dist[triplets[:, 0], triplets[:, 1]].squeeze()
    d_ik = dist[triplets[:, 0], triplets[:, 2]].squeeze()
    return (d_ij - d_ik).clamp(min=0).sum() + lambd * kernel_matrix.trace()


def _gnmds_x_loss_torch(embedding, triplets, lambd):
    """ Equation (1) of Terada & Luxburg (2014) """
    import torch  # Pytorch is an optional dependency

    X = embedding[triplets.long()]
    anchor, positive, negative = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    triplet_loss = torch.nn.functional.triplet_margin_loss(anchor, positive, negative,
                                                           margin=lambd, p=2, reduction='none')
    return (triplet_loss**2).sum()
