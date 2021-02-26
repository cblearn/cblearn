from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np
import scipy

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.utils import assert_torch_is_available  # , torch_minimize_lbfgs
from cblearn.embedding import _torch_utils


class FORTE(BaseEstimator, TripletEmbeddingMixin):
    """ Generalized Non-metric Multidimensional Scaling (FORTE).

        FORTE [1]_ minimizes a kernel version of the triplet hinge soft objective as a smooth relaxation of the triplet error.

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
        >>> estimator = FORTE(n_components=2)
        >>> embedding = estimator.fit_transform(triplets)
        >>> embedding.shape
        (15, 2)
        >>> round(estimator.score(triplets), 1)
        1.0


        References
        ----------
        .. [1] Terada, Y., & Luxburg, U. (2014). Local ordinal embedding.
               International Conference on Machine Learning, 847–855.
        .. [2] Vankadara, L. et al. (2019) Insights into Ordinal Embedding Algorithms: A Systematic Evaluation
               Arxiv Preprint, https://arxiv.org/abs/1912.01666
        """

    def __init__(self, n_components=2, margin=1, max_iter=2000, learning_rate=1000, batch_size=1000000, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None,
                 algorithm: str = "LineSearch", device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            margin:
                Scale parameter which only takes strictly positive value.
                Defines the intended minimal difference of distances in the embedding space between
                for any triplet.
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
        self.margin = margin
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.algorithm = algorithm
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, X: utils.Questions, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'FORTE':
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

        if self.algorithm == "LineSearch":
            assert_torch_is_available()
            result = self.torch_minimize_kernel(init, triplets.astype(int), device=self.device,
                                                max_iter=self.max_iter, batch_size=self.batch_size)
        else:
            raise ValueError(f"Unknown FORTE algorithm '{self.algorithm}'. Try 'K' or 'X' instead.")

        if self.verbose and not result.success:
            print(f"FORTE's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self


    def torch_minimize_kernel(self, init: np.ndarray, triplets: np.ndarray, device: str, max_iter: int, batch_size: int
                              ) -> 'scipy.optimize.OptimizeResult':
        """ Pytorch minimization routine using LineSearch.

            This function aims to be a pytorch version of :func:`scipy.optimize.minimize`.

            Args:
                init:
                    The initial parameter values.
                args:
                    Sequence of additional arguments, passed to the objective.
                device:
                    Device to run the minimization on, usually "cpu" or "cuda".
                    "auto" uses "cuda", if available.
                max_iter:
                    The maximum number of optimizer iteration.
            Returns:
                Dict-like object containing status and result information
                of the optimization.
        """
        import torch  # pytorch is an optional dependency of the library
        from torch.autograd import grad


        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        triplet_num = triplets.shape[0]
        learning_rate = torch.tensor([self.learning_rate]).to(device)  # learning rate
        triplets = torch.tensor(triplets).to(device).long()
        batches = 1 if batch_size > triplet_num else triplet_num // batch_size
        rho = torch.tensor([0.5]).to(device)  # backtracking line search parameter
        c1 = torch.tensor([0.0001]).to(device)  # Amarijo stopping condition parameter
        X = torch.tensor(init, dtype=torch.float).to(device)
        K = torch.mm(X, torch.transpose(X, 0, 1)).to(device) * .1
        # factr = 1e7 * np.finfo(float).eps

        loss = float("inf")
        success, message = True, ""
        for n_iter in range(max_iter):
            epoch_loss = 0
            for batch_ind in range(batches):
                batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
                K.requires_grad = True
                old_loss = _torch_forte_loss(K, batch_trips)
                epoch_loss += old_loss.item()
                gradient = grad(old_loss, K)[0]
                K.requires_grad = False
                new_K = _torch_utils.project_rank(K - learning_rate * gradient, self.n_components)
                new_loss = _torch_forte_loss(new_K, batch_trips)
                diff = new_K - K
                beta = rho
                norm_grad = torch.norm(gradient, dim=0)
                norm_grad_sq = torch.sum(norm_grad)
                inner_t = 0
                while new_loss > old_loss - c1 * learning_rate * norm_grad_sq and inner_t < 10:
                    beta = beta * beta
                    new_loss = _torch_forte_loss(K + beta * diff, batch_trips)
                    inner_t += 1
                if inner_t > 0:
                    learning_rate = torch.max(torch.FloatTensor([0.1]).to(device), learning_rate * rho)

                K = _torch_utils.project_rank(K + beta * diff, self.n_components)

            # prev_loss = loss
            loss = epoch_loss / triplets.shape[0]

        # SVD to get embedding
        U, s, _ = K.svd()
        X = U[:, :self.n_components].mm(s[:self.n_components].sqrt().diag())
        return scipy.optimize.OptimizeResult(
            x=X.cpu().detach().numpy(), fun=loss, nit=n_iter,
            success=success, message=message)


def _torch_forte_loss(kernel_matrix, triplets):
    diag = kernel_matrix.diag()[:, None]
    dist = -2 * kernel_matrix + diag + diag.transpose(0, 1)
    d_ij = dist[triplets[:, 0], triplets[:, 1]].squeeze()
    d_ik = dist[triplets[:, 0], triplets[:, 2]].squeeze()
    return (1 + (d_ij - d_ik).exp()).log().sum()
