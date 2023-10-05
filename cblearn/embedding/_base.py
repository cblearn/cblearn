from typing import Optional
from numpy.typing import ArrayLike

import numpy as np
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d
from scipy.special import expit
from sklearn.metrics import pairwise
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

import cblearn as cbl
from cblearn import Comparison
from cblearn.embedding import _torch_utils


class QuadrupletEmbeddingMixin(TransformerMixin, ClassifierMixin):
    def __init__(self, n_components,
                 backend='scipy',
                 verbose=0,
                 max_iter=1000,
                 random_state=None,
                 batch_size=50000,
                 learning_rate=1,
                 torch_kernel_kwargs={},
                 device='auto',
                 use_quadruplets=False, n_init=1,
                 kernel=False,
                 torch_optimizer='adam'):
        """
        Args:
            use_quadruplets: Automatically transform triplets to quadruplets.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.batch_size = batch_size
        self.use_quadruplets = use_quadruplets
        self.learning_rate = learning_rate
        self.torch_kernel_kwargs = torch_kernel_kwargs
        self.n_init = n_init
        self.backend = backend
        self.kernel = kernel
        self.torch_optimizer = torch_optimizer

    def _more_tags(self):
        tags = {
            'requires_positive_X': True,
            'requires_positive_y': False,
            'binary_only': True,
            'preserves_dtype': [],  # transform(X) does not preserve dtype
            'poor_score': True,  # non-triplet inputs are not meaningful
            'X_types': ['triplets', '2darray']  # 2darray is not true, but required to run sklearn tests
        }
        if self.use_quadruplets:
            tags['X_types'].append('quadruplets')
        return tags

    def _prepare_data(self, X: Comparison, y: ArrayLike, quadruplets=False, return_y=True,
                      sample_weight=None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Validate `X` and `y` and binarize `y`.

        Args:
            X: Training data.
            y: Target values.

        Returns:
            X: (n_samples, n_features), Validated training data.
            y: (n_samples,), Validated target values.
        """
        sample_weight = _check_sample_weight(sample_weight, X)

        if y is None:
            self.classes_ = np.array([-1, 1])
        else:
            y = column_or_1d(y, warn=True)
            self.classes_, y = np.unique(y, return_inverse=True)
            y = np.array([-1, 1])[y]
        if len(self.classes_) < 2:
            raise ValueError(
                "This solver needs samples of 2 classes"
                " in the data, but the data contains only one"
                " class: %r" % self.classes_[0]
            )

        if quadruplets:
            result = cbl.check_quadruplets(X, y, return_y=return_y)
        else:
            result = cbl.check_triplets(X, y, return_y=return_y)

        if sample_weight is None:
            return result
        else:
            return (result, sample_weight)

    def transform(self, X: Comparison = None, y: Optional[ArrayLike] = None) -> np.ndarray:
        check_is_fitted(self, 'embedding_')
        return self.embedding_

    def decision_function(self, X: Comparison) -> np.ndarray:
        check_is_fitted(self, 'embedding_')
        X = cbl.check_quadruplets(X, return_y=False, canonical=False)
        X = self.embedding_[X]
        near_distance = pairwise.paired_euclidean_distances(X[:, 0], X[:, 1])
        far_distance = pairwise.paired_euclidean_distances(X[:, 2], X[:, 3])
        return far_distance - near_distance

    def predict(self, X: Comparison) -> np.ndarray:
        scores = self.decision_function(X)

        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = np.argmax(scores, axis=1)

        return np.take(self.classes_, indices, axis=0)

    def predict_proba(self, X):
        """Probability estimation.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        return np.vstack([1 - prob, prob]).T

    def score(self, X: Comparison, y: ArrayLike = None, sample_weight=None) -> float:
        """ Triplet score on the estimated embedding.

        Args:
            X: Triplet or quadruplet comparisons.
            y: Binary responses {-1, 1}.
            sample_weight: Individual weights for each sample.
        Returns.
            Fraction of correct triplets.
        """
        X, y = cbl.check_quadruplets(X, y, return_y=True)
        return ClassifierMixin.score(self, X, y, sample_weight=sample_weight)

    def _scipy_loss(self, embedding, quads, D_small, D_large, *args):
        raise NotImplementedError("""This estimator does not support scipy optimization.
                                     Try using backend='torch' instead.""")

    def _torch_loss(self, embedding, quads, D_small, D_large, *args):
        raise NotImplementedError("""This estimator does not support torch optimization.
                                     Try using backend='scipy' instead.""")

    def _torch_kernel_loss(self, kernel, quads):
        raise NotImplementedError("""This estimator does not support torch kernel optimization.
                                     Try using kernel=False or backend='scipy' instead.""")

    def fit(self, X: Comparison, y: np.ndarray = None,
            sample_weight: Optional[np.ndarray] = None,
            init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'QuadrupletEmbeddingMixin':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            sample_weight: Optional weight per triplet.
            init: Initial embedding for optimization.
                  Pass a list to run the optimization multiple times and
                  return the best result.
            n_objects: Number of objects in the embedding. If none, infer from X.
        Returns:
            self.
        """
        if sample_weight is None:
            sample_weight = 1
        X, sample_weight = self._prepare_data(
            X, y, return_y=False,
            quadruplets=self.use_quadruplets,
            sample_weight=sample_weight)

        if not n_objects:
            n_objects = X.max() + 1
        random_state = check_random_state(self.random_state)
        if init is None:
            inits = [random_state.multivariate_normal(np.zeros(self.n_components),
                     np.eye(self.n_components), size=n_objects) for _ in range(self.n_init)]
        else:
            init = np.array(init)
            if init.ndim == 3:
                inits = init
            else:
                inits = [init]

        best_result = None
        for init in inits:
            if self.backend == "torch":
                _torch_utils.assert_torch_is_available()

                if self.kernel:
                    result = _torch_utils.torch_minimize_kernel(
                        self.torch_optimizer,
                        self._torch_kernel_loss, init, data=[X.astype(int)], args=(sample_weight,),
                        device=self.device, max_iter=self.max_iter, batch_size=self.batch_size, lr=self.learning_rate,
                        seed=random_state.randint(1), **self.torch_kernel_kwargs)
                else:
                    def _loss(embedding, quads, *args):
                        import torch
                        embedding = embedding[quads.long()]
                        D_near = torch.pairwise_distance(embedding[:, 0, :], embedding[:, 1, :], p=2)
                        if quads.shape[1] == 3:  # is a triplet
                            D_far = torch.pairwise_distance(embedding[:, 0, :], embedding[:, 2, :], p=2)
                        else:
                            D_far = torch.pairwise_distance(embedding[:, 2, :], embedding[:, 3, :], p=2)
                        loss = self._torch_loss(embedding, quads, D_near, D_far, *args)
                        return loss

                    result = _torch_utils.torch_minimize(
                        self.torch_optimizer, _loss, init, data=(X,), args=(sample_weight,),
                        device=self.device, max_iter=self.max_iter,
                        lr=self.learning_rate, seed=random_state.randint(1))
            elif self.backend == "scipy":
                if self.kernel:
                    raise NotImplementedError("""
                        This estimator does not support scipy kernel optimization.
                        Try using kernel=False or backend='torch' instead.""")

                def _loss(embedding, embedding_shape, quadruplet, *args):
                    embedding = embedding.reshape(embedding_shape)
                    if embedding.ndim == 3:  # multible embedding algorithm
                        dist = np.asarray(
                            [distance_matrix(e, e, p=2)
                             for e in embedding])
                    else:
                        dist = distance_matrix(embedding, embedding, p=2)

                    ij_dist = dist[..., quadruplet[:, 0], quadruplet[:, 1]]
                    if quadruplet.shape[1] == 3:  # is a triplet
                        kl_dist = dist[..., quadruplet[:, 0], quadruplet[:, 2]]
                    else:
                        kl_dist = dist[..., quadruplet[:, 2], quadruplet[:, 3]]
                    loss, grad = self._scipy_loss(
                        embedding, quadruplet, ij_dist, kl_dist, *args)
                    return loss, grad.ravel()

                result = minimize(_loss, init.ravel(), args=(init.shape, X, sample_weight),
                                  method='L-BFGS-B', jac=True, options=dict(maxiter=self.max_iter, disp=self.verbose))

            else:
                raise ValueError(f"Unknown backend '{self.backend}'. Try 'scipy' or 'torch' instead.")

            if best_result is None or best_result.fun > result.fun:
                best_result = result
            if self.verbose and not result.success:
                print(f"Embedding optimization failed: {result.message}.\n"
                      f"{'Retry with another initialization...' if init != inits[-1] else ''}")

        self.embedding_ = best_result.x.reshape(-1, self.n_components)
        self.stress_, self.n_iter_ = best_result.fun, best_result.nit
        self.optimize_result_ = best_result
        return self