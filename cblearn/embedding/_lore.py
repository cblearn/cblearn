"""LORE: Low-Rank Ordinal Embedding estimator for CBlearn."""

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding._lore_utils import _lore_train_torch, _lore_train_scipy


class LORE(TripletEmbeddingMixin, BaseEstimator):
    """LORE: Jointly Learning the Intrinsic Dimensionality and Relative
    Similarity Structure from Ordinal Data.

    LORE [1]_ learns an ordinal embedding and simultaneously estimates its
    intrinsic dimensionality by applying a Schatten-p (or nuclear-norm)
    regularisation to the singular values of the embedding matrix.  The
    optimisation is a proximal gradient method: at each step the embedding
    is updated via gradient descent on the logistic triplet loss, followed
    by a singular-value thresholding operation that drives low-information
    dimensions toward zero.

    Unlike fixed-rank estimators (SOE, CKL, ...), LORE starts with an
    over-estimated ambient dimension ``n_components`` and lets the
    regularisation discover the true rank.  The estimated rank is exposed
    through the ``rank_`` attribute after fitting.

    Two backends are supported:

    * **torch** (default) - uses the reference PyTorch implementation with
      autograd-based Lipschitz-constant estimation via the power method.
      Supports CUDA GPUs.  Requires the ``torch`` extra
      (``pip install cblearn[torch]``).
    * **scipy** - a pure NumPy/SciPy re-implementation that requires no
      GPU or PyTorch installation.

    Attributes:
        embedding_ (np.ndarray): Learned embedding of shape
            ``(n_objects, n_components)``.  Columns corresponding to
            zero-valued singular values are numerically near zero.
        rank_ (int): Estimated intrinsic dimensionality, i.e. the number of
            non-zero singular values in the final embedding.
        stress_ (float): Final objective value (triplet loss + regularisation).
        n_iter_ (int): Number of optimisation iterations performed.
        sigma_history_ (list): Singular-value arrays recorded at each
            iteration.
        objectives_ (list): Objective values recorded at each iteration.

    Examples:

    >>> from cblearn import datasets
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> true_embedding = rng.rand(15, 2)
    >>> triplets = datasets.make_random_triplets(true_embedding, result_format='list-order',
    ...                                          size=500, random_state=rng)
    >>> estimator = LORE(n_components=5, backend="scipy", max_iter=200, random_state=42)
    >>> embedding = estimator.fit_transform(triplets, n_objects=15)
    >>> embedding.shape
    (15, 5)
    >>> estimator.rank_ <= 5
    True
    >>> round(estimator.score(triplets), 1) >= 0.5
    True

    References:
        .. [1] Anand, V., Helbling, A., Davenport, M. A., Berman, G. J.,
               Alagapan, S., & Rozell, C. J. (2026). LORE: Jointly Learning
               The Intrinsic Dimensionality and Relative Similarity Structure
               from Ordinal Data. International Conference on Learning
               Representations (ICLR). https://arxiv.org/abs/2602.04192
    """

    def __init__(
        self,
        n_components: int = 2,
        lamb: float = 0.01,
        p: float = 0.5,
        margin: float = 0.1,
        mu: Union[str, float] = "default",
        verbose: bool = False,
        random_state: Union[None, int, np.random.RandomState] = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        zero: float = 1e-15,
        backend: str = "torch",
        device: str = "auto",
    ):
        """Initialise the LORE estimator.

        Args:
            n_components: Ambient (over-estimated) embedding dimension.
                LORE will discover the true intrinsic rank via regularisation.
            lamb: Regularisation strength lambda (default 0.01, as used in the
                paper).  Larger values promote lower rank at the cost of
                higher triplet loss.
            p: Schatten-p exponent.  ``p=1.0`` uses the nuclear norm (convex);
                ``p<1`` uses the non-convex Schatten-p quasi-norm (default 0.5)
                which promotes sparser singular-value spectra.
            margin: Margin for the logistic triplet loss.
            mu: Lipschitz constant used as the gradient step size.  When set
                to ``"default"`` the torch backend estimates it via the power
                method (autograd) and the scipy backend uses a conservative
                estimation.  Pass a positive float to override.
            verbose: If True, print per-iteration progress.
            random_state: Seed for reproducible random initialisation.
            max_iter: Maximum number of proximal gradient iterations.
            tol: Convergence tolerance on the change in objective value.
            zero: Singular values below this threshold are set to zero.
            backend: Optimisation backend.  ``"torch"`` (default) uses the
                reference PyTorch implementation with GPU support.
                ``"scipy"`` uses a pure NumPy/SciPy implementation.
            device: Device for the torch backend.  ``"auto"`` selects CUDA
                when available, otherwise CPU.  Ignored for ``backend="scipy"``.
        """
        self.n_components = n_components
        self.lamb = lamb
        self.p = p
        self.margin = margin
        self.mu = mu
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.zero = zero
        self.backend = backend
        self.device = device

    def fit(
        self,
        X: utils.Query,
        y: Optional[np.ndarray] = None,
        init: Optional[np.ndarray] = None,
        n_objects: Optional[int] = None,
    ) -> 'LORE':
        """Fit the LORE embedding to triplet comparison data.

        Args:
            X: Triplet comparisons in any CBlearn query format.  After
                conversion to ``list-order`` format the array has shape
                ``(n_triplets, 3)`` where each row is
                ``[anchor, closer_object, farther_object]``.
            y: Ignored; present for API compatibility.
            init: Optional initial embedding of shape
                ``(n_objects, n_components)``.  If None a random Gaussian
                initialisation is used.
            n_objects: Number of objects in the embedding.  Inferred from X
                when not provided.

        Returns:
            self
        """
        self.fit_X_ = utils.check_query(X, result_format='list-order')
        queries = utils.check_query_response(X, y, result_format='list-order')
        self.n_features_in_ = 3

        if n_objects is None:
            n_objects = int(queries.max()) + 1

        random_state = check_random_state(self.random_state)

        if init is None:
            init = random_state.multivariate_normal(
                np.zeros(self.n_components),
                np.eye(self.n_components),
                size=n_objects,
            )
        else:
            init = np.array(init, dtype=float)

        seed = int(random_state.randint(2 ** 31))

        if self.backend == "torch":
            from cblearn.embedding._torch_utils import assert_torch_is_available
            assert_torch_is_available()
            result = _lore_train_torch(
                init=init,
                triplets=queries.astype(int),
                lamb=self.lamb,
                p=self.p,
                margin=self.margin,
                mu=self.mu,
                max_iter=self.max_iter,
                tol=self.tol,
                zero=self.zero,
                seed=seed,
                verbose=self.verbose,
                device=self.device,
            )
        elif self.backend == "scipy":
            result = _lore_train_scipy(
                init=init,
                triplets=queries.astype(int),
                lamb=self.lamb,
                p=self.p,
                margin=self.margin,
                mu=self.mu,
                max_iter=self.max_iter,
                tol=self.tol,
                zero=self.zero,
                seed=seed,
                verbose=self.verbose,
            )
        else:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                "Valid options are 'torch' and 'scipy'."
            )

        self.embedding_ = result.x
        self.stress_ = result.fun
        self.n_iter_ = result.nit
        self.rank_ = result.ranks[-1] if result.ranks else self.n_components
        self.sigma_history_ = result.sigma_history
        self.objectives_ = result.objectives

        return self
