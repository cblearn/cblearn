from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np
import scipy
# from scipy.optimize import minimize
from scipy.spatial import distance_matrix

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.utils import assert_torch_is_available, torch_minimize_lbfgs


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
        >>> true_embedding = np.random.rand(15, 2)
        >>> triplets = datasets.make_random_triplets(true_embedding, result_format='list-order', size=1000)
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
        >>> estimator = GNMDS(n_components=2, random_state=42, algorithm='K')
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> estimator.score(triplets) > 0.8
        True

        The following is running on the CUDA GPU, if available (but requires pytorch installed).

        >>> estimator = GNMDS(n_components=2, algorithm="X", random_state=42)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> estimator.score(triplets) > 0.8
        True

        References
        ----------
        .. [1] Terada, Y., & Luxburg, U. (2014). Local ordinal embedding.
               International Conference on Machine Learning, 847–855.
        .. [2] Vankadara, L. et al. (2019) Insights into Ordinal Embedding Algorithms: A Systematic Evaluation
               Arxiv Preprint, https://arxiv.org/abs/1912.01666
        """

    def __init__(self, n_components=2, margin=1, max_iter=2000, C=0.01, learning_rate=100, batch_size=1000000, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None,
                 algorithm: str = "X", device: str = "auto"):
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
        self.C = C
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
            assert_torch_is_available()
            result = self.torch_minimize_adam_kernel(init, triplets.astype(int), device=self.device,
                                                     max_iter=self.max_iter, batch_size=self.batch_size)
        elif self.algorithm == "X":
            """
                The embedding (X) version of the algorithm.
            """
            assert_torch_is_available()
            result = torch_minimize_lbfgs(_gnmds_x_loss_torch, init, args=(triplets.astype(int), self.margin),
                                          device=self.device, max_iter=self.max_iter)
        else:
            raise ValueError(f"Unknown GNMDS algorithm '{self.algorithm}'. Try 'K' or 'X' instead.")

        if self.verbose and not result.success:
            print(f"GNMDS's optimization failed with reason: {result.message}.")
        self.embedding_ = result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = result.fun, result.nit
        return self

    def torch_minimize_adam_kernel(self, init: np.ndarray, triplets: np.ndarray, device: str, max_iter: int, batch_size: int
                                   ) -> 'scipy.optimize.OptimizeResult':
        """ Pytorch minimization routine using Adam.

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

        def _gnmds_kernel_loss_torch(K, triplets, C):

            diag = torch.diag(K)[:, None]
            Dist = -2 * K + diag + torch.transpose(diag, 0, 1)
            loss = _gnmds_k_hinge(Dist[triplets[:, 0], triplets[:, 1]].squeeze(),
                                  Dist[triplets[:, 0], triplets[:, 2]].squeeze())

            return torch.sum(loss) + C * torch.trace(K)

        def _gnmds_k_hinge(d_ij, d_ik):
            device = d_ij.device
            loss = torch.max(d_ij - d_ik + 1, torch.FloatTensor([0]).to(device))
            return loss

        def _project_rank(K, dim):
            D, U = torch.symeig(K, eigenvectors=True)  # will K be surely symmetric?
            D = torch.max(D[-dim:], torch.Tensor([0.]).to(K.device))
            return torch.mm(torch.mm(U[:, -dim:], torch.diag(D)), torch.transpose(U[:, -dim:], 0, 1))

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        triplet_num = triplets.shape[0]
        triplets = torch.tensor(triplets).to(device).long()
        batches = 1 if batch_size > triplet_num else triplet_num // batch_size
        C = torch.Tensor([self.C]).to(device)
        X = torch.tensor(init, dtype=torch.float).to(device)
        K = torch.mm(X, torch.transpose(X, 0, 1)).to(device) * .1
        # factr = 1e7 * np.finfo(float).eps

        optimizer = torch.optim.Adam(params=[K], lr=self.learning_rate, amsgrad=True)
        loss = float("inf")
        success, message = True, ""
        for n_iter in range(max_iter):
            epoch_loss = 0
            for batch_ind in range(batches):
                batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ]  # a batch of triplets
                K.requires_grad = True

                batch_loss = _gnmds_kernel_loss_torch(K, batch_trips, C)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                K.requires_grad = False
                epoch_loss += batch_loss.item()
                # projection back onto semidefinite cone
                K = _project_rank(K, self.n_components)

            # prev_loss = loss
            loss = epoch_loss / triplets.shape[0]

        # SVD to get embedding
        U, s, _ = torch.svd(K)
        X = torch.mm(U[:, :self.n_components], torch.diag(torch.sqrt(s[:self.n_components])))
        return scipy.optimize.OptimizeResult(
            x=X.cpu().detach().numpy(), fun=loss, nit=n_iter,
            success=success, message=message)


def _gnmds_x_loss_torch(embedding, triplets, margin):
    """ Equation (1) of Terada & Luxburg (2014) """
    import torch  # Pytorch is an optional dependency

    X = embedding[triplets]
    anchor, positive, negative = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    triplet_loss = torch.nn.functional.triplet_margin_loss(anchor, positive, negative,
                                                           margin=margin, p=2, reduction='none')
    return torch.sum(triplet_loss**2)


def _gnmds_x_loss(x, x_shape, triplets, margin):
    """ Equation (1) of Terada & Luxburg (2014) """
    X = x.reshape(x_shape)
    X_dist = distance_matrix(X, X)
    ij_dist = X_dist[triplets[:, 0], triplets[:, 1]]
    kl_dist = X_dist[triplets[:, 0], triplets[:, 2]]
    stress = np.maximum(ij_dist + margin - kl_dist, 0) ** 2
    return stress.sum()