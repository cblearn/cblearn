from scipy.spatial import distance

from typing import Optional, Union

from sklearn.base import BaseEstimator
import numpy as np

import cblearn as cbl
from cblearn import Comparison
from cblearn.embedding._base import QuadrupletEmbeddingMixin


EPS = np.finfo(float).tiny


class BaseSTE(BaseEstimator, QuadrupletEmbeddingMixin):
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


        References
        ----------
        .. [1] van der Maaten, L., & Weinberger, K. (2012). Stochastic triplet embedding.
               2012 IEEE International Workshop on Machine Learning for Signal Processing, 1–6.
        .. [2] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, heavy_tailed=False, verbose=False, lambd=0,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy", learning_rate=0.0002, batch_size=50_000,  device: str = "auto"):
        """ Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            heavy_tailed:
                If false, STE is using the Gaussian kernel,
                If true, t-STE is using the heavy-tailed student-t kernel.
            verbose: boolean, default=False
                Enable verbose output.
            lambd: float, Amount of L2 regularization.
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
        self.heavy_tailed = heavy_tailed
        self.lambd = lambd
        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate,
            backend=backend,
            use_quadruplets=False)

    def _torch_loss(self, embedding, triplets, dist_pos, dist_neg,
                    sample_weights, *args):
        dof = max(embedding.shape[1] - 1, 1)
        dist_1 = dist_pos**2
        dist_2 = dist_neg**2
        if self.heavy_tailed:
            t_dist_1 = (1 + dist_1 / 2.)**(-(dof + 1) / 2)
            p = t_dist_1 / (t_dist_1 + (1 + dist_2 / 2.)**(-(dof + 1) / 2) + 1e-16)
        else:
            p = (-dist_1).exp() / ((-dist_1).exp() + (-dist_2).exp() + 1e-16)

        return -(sample_weights * p).log().sum() + self.lambd * embedding.pow(2).sum()

    def _scipy_loss(self, X, triplets, dist_pos, dist_neg, sample_weights, *args):
        """ Calculates the log STE loss"""
        n_objects, n_dim = X.shape
        dof = max(n_dim - 1, 1)
        if self.heavy_tailed:
            base_dist_pos = 1 + dist_pos**2 / dof
            base_dist_neg = 1 + dist_neg**2 / dof
            dist_pos = base_dist_pos**((dof + 1) / -2)
            dist_neg = base_dist_neg**((dof + 1) / -2)
        else:
            dist_pos = np.exp(-dist_pos**2)
            dist_neg = np.exp(-dist_neg**2)

        I, J, K = tuple(triplets.T)
        P = dist_pos / np.maximum(dist_pos + dist_neg, EPS)
        loss = -np.mean(sample_weights * np.log(np.maximum(P, EPS))) + self.lambd * np.mean(X**2)

        if self.heavy_tailed:
            dist_pos_inv = (1 / base_dist_pos)[..., np.newaxis]
            dist_neg_inv = (1 / base_dist_neg)[..., np.newaxis]
            grad_triplets = - (dof + 1) / dof * np.array([
                dist_pos_inv * (X[I] - X[J]) - dist_neg_inv * (X[I] - X[K]),
                - dist_pos_inv * (X[I] - X[J]),
                dist_neg_inv * (X[I] - X[K])])
        else:
            grad_triplets = - 2 * np.array([
                (X[I] - X[J]) - (X[I] - X[K]),
                - (X[I] - X[J]),
                (X[I] - X[K])])

        grad_triplets *= (sample_weights * (1 - P))[np.newaxis, :, np.newaxis]

        loss_grad = np.empty_like(X)
        for dim in range(X.shape[1]):
            loss_grad[:, dim] = np.bincount(triplets[:, 0], grad_triplets[0, :, dim], n_objects)
            loss_grad[:, dim] += np.bincount(triplets[:, 1], grad_triplets[1, :, dim], n_objects)
            loss_grad[:, dim] += np.bincount(triplets[:, 2], grad_triplets[2, :, dim], n_objects)
        loss_grad = loss_grad / len(triplets)  # this normalizes the gradient
        loss_grad = -loss_grad + 2 * self.lambd * X

        return loss, loss_grad.ravel()


class STE(BaseSTE):
    """ Stochastic Triplet Embedding (STE)

    The "classic" variant of :class:`BaseSTE`, that assumes normal distributed distances.

    Examples:

        >>> from cblearn import datasets
        >>> seed = np.random.RandomState(40)
        >>> true_embedding = seed.rand(15, 2)
        >>> triplets = datasets.make_random_triplets(true_embedding, result_format='list-order',
        ...                                          size=1000, random_state=seed)
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
        >>> estimator = STE(n_components=2, random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> estimator.score(triplets) > 0.9
        True

        The following is running on the CUDA GPU, if available (but requires pytorch installed).

        >>> estimator = STE(n_components=2, backend="torch", random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> estimator.score(triplets) > 0.9
        True
    """
    def __init__(self, n_components=2, verbose=False, lambd=0,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy", learning_rate=1, batch_size=50_000,  device: str = "auto"):
        heavy_tailed = False
        return super().__init__(n_components, heavy_tailed, verbose, lambd, random_state, max_iter, backend,
                                learning_rate, batch_size, device)


class TSTE(BaseSTE):
    """ t-Distributed Stochastic Triplet Embedding (t-STE)

    Variant of :class:`BaseSTE`, that assumes t-student distributed distances
    which leads to better optimization properties.

    Examples:

        >>> from cblearn import datasets
        >>> seed = np.random.RandomState(40)
        >>> true_embedding = seed.rand(15, 2)
        >>> triplets = datasets.make_random_triplets(true_embedding, result_format='list-order',
        ...                                          size=1000, random_state=seed)
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
        >>> estimator = TSTE(n_components=2, random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> estimator.score(triplets) > 0.9
        True

        The following is running on the CUDA GPU, if available (but requires pytorch installed).

        >>> estimator = TSTE(n_components=2, backend="torch", random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> estimator.score(triplets) > 0.9
        True
    """
    def __init__(self, n_components=2, verbose=False, lambd=0,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy", learning_rate=1, batch_size=50_000,  device: str = "auto"):
        heavy_tailed = True
        return super().__init__(n_components, heavy_tailed, verbose, lambd, random_state, max_iter, backend,
                                learning_rate, batch_size, device)


class MVTE(BaseSTE):
    """ Multi-view Triplet Embedding (MVTE)

    Variant of :class:`BaseSTE`, that fits multiple so-called views of an embedding.

    Examples:

        >>> import cblearn as cbl
        >>> from cblearn import datasets
        >>> seed = np.random.RandomState(40)
        >>> true_embedding = seed.rand(2, 15, 1)
        >>> X1, y1 = datasets.make_random_triplets(true_embedding[0], size=1000, result_format='list-count')
        >>> X2, y2 = datasets.make_random_triplets(true_embedding[1], size=1000, result_format='list-count')
        >>> X, y = np.r_[X1, X2], np.r_[y1, y2]
        >>> estimator = MVTE(n_components=1, n_maps=2, random_state=seed)
        >>> estimator.fit(X, y).embedding_.shape
        (2, 15, 1)
        >>> estimator.score(X, y) > 0.8
        True

        >>> from sklearn.model_selection import cross_val_score
        >>> X, y = cbl.check_triplets(datasets.fetch_vogue_cover_similarity().triplet)
        >>> estimator = MVTE(n_components=1, n_maps=3, random_state=seed)
        >>> cross_val_score(estimator, X, y, cv=10).mean() > 0.95  # see Fig 3c in [1]
        True

    References
    ----------
    [1] Ehsan Amid, Antti Ukkonen, Multiview Triplet Embedding: Learning Attributes in Multiple Maps,
        Proceedings of the 32nd International Conference on Machine Learning, PMLR
    """
    def __init__(self, n_components=2, n_maps=1, heavy_tailed=True, verbose=False, lambd=0,
                 random_state: Union[None, int, np.random.RandomState] = None, max_iter=1000,
                 backend: str = "scipy", learning_rate=1, batch_size=50_000,  device: str = "auto"):
        self.n_maps = n_maps
        return super().__init__(n_components, heavy_tailed, verbose, lambd, random_state, max_iter, backend,
                                learning_rate, batch_size, device)

    def _scipy_loss(self, X, triplets, dist_pos, dist_neg, sample_weights, *args):
        ratio = satifyability_ratio(X, triplets)
        ratio[ratio <= 1] = EPS
        ratio[np.isinf(ratio)] = 1e5
        z = ratio / ratio.sum(axis=0)
        cost = 0
        grad = np.zeros_like(X)
        for i in range(len(z)):
            c, g = BaseSTE._scipy_loss(self, X[i], triplets,
                                       dist_pos[i], dist_neg[i],
                                       sample_weights * z[i], *args)
            cost += c
            grad[i, :, :] = g.reshape(X.shape[1:])
        return cost, grad.ravel()

    def satifyability_ratio(self, triplets, metric='euclidean'):
        """ Distance ratio of triplets for multiple embeddings.

        d_far / d_close

        Amid & Ukkonen (2015), Eq. 7 """
        return satifyability_ratio(self.embedding_, triplets, metric)

    def fit(self, X: Comparison, y: np.ndarray = None,
            init: np.ndarray = None,
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
        X = cbl.check_triplets(X, y, return_y=False)
        if n_objects is None:
            n_objects = X.max() + 1
        BaseSTE.fit(self, X, y=None, init=init, n_objects=(self.n_maps, n_objects))
        self.embedding_ = self.embedding_.reshape(self.n_maps, n_objects, self.n_components)
        return self

    def predict(self, X: Comparison) -> np.ndarray:
        all_embeddings = self.embedding_
        predictions = np.empty((len(all_embeddings), len(X)), dtype=int)
        for i, embedding in enumerate(all_embeddings):
            self.embedding_ = embedding
            predictions[i, :] = BaseSTE.predict(self, X)
        self.embedding_ = all_embeddings
        return predictions

    def score(self, X: Comparison, y: np.ndarray = None):
        X, y = cbl.check_triplets(X, y, return_y=True)
        pred = self.predict(X)
        return (pred == y[np.newaxis, :]).any(axis=0).mean()


def satifyability_ratio(embeddings, triplets, metric='euclidean'):
    """ Distance ratio of triplets for multiple embeddings.

    d_far / d_close

    Amid & Ukkonen (2015), Eq. 7 """
    m, n, d = embeddings.shape
    ratios = np.empty((m, len(triplets)))
    for i, embedding in enumerate(embeddings):
        D = distance.squareform(distance.pdist(embedding, metric))
        ratios[i, :] = (D[triplets[:, 0], triplets[:, 2]]
                        / D[triplets[:, 0], triplets[:, 1]])
    return ratios