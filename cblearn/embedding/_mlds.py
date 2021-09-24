from typing import Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from scipy.optimize import minimize

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin


class MLDS(BaseEstimator, TripletEmbeddingMixin):
    """ A maximum-likelihood difference scaling (MLDS) estimator .

    MLDS [1]_ is limited to monotonic, one-dimensional embeddings.

    note::
        This method assumes, that the objects can be embedded in a one-dimensional space
        and that the object indices are consistent to their order in this space.

    Attributes:
        embedding_: The final embedding, shape (n_objects, 1)
        log_likelihood_: The final log-likelihood of the embedding.
        n_iter_: Optimization iterations


    >>> from cblearn import datasets
    >>> true_embedding = sorted(np.random.uniform(1, 2, (15, 1)))
    >>> triplets = datasets.make_random_triplets(true_embedding, size=400, monotonic=True, result_format='list-order')
    >>> triplets.shape, np.unique(triplets).shape
    ((400, 3), (15,))
    >>> estimator = MLDS(random_state=42).fit(triplets)
    >>> estimator.embedding_.shape
    (15, 1)
    >>> estimator.score(triplets) > 0.9
    True
    >>> estimator = MLDS(method='optim', random_state=42).fit(triplets)
    >>> estimator.score(triplets) > 0.9
    True


    References
    ----------
    .. [1] M Knoblauch, K., & Maloney, L. T. (2012). Modeling Psychophysical Data in R.
           Springer New York. https://doi.org/10.1007/978-1-4614-4475-6
    """

    def __init__(self, n_components: int = 1, random_state: Union[None, int, np.random.RandomState] = None,
                 method='glm', verbose: int = 0, max_iter: int = 1000):
        """
        Args:
            n_components: Embedding dimension for api compatibility. Only 1 is supported for MLDS.
            random_state: The seed of the pseudo random number generator used to initialize the optimization.
            method: Optimizer method, either 'glm' or 'optim'.
            verbose: Enable verbose output.
            max_iter: Maximum number of optimization iterations.
        """
        if n_components != 1:
            raise ValueError(f"MLDS expects n_components=1, got {n_components}")
        self.n_components = n_components
        self.random_state = random_state
        self.method = method
        self.verbose = verbose
        self.max_iter = max_iter

    def _log_likelihood(self, x, quadruplet, answer, float_min=np.finfo(float).tiny):
        prob = norm.cdf((x[quadruplet[:, 0]] - x[quadruplet[:, 1]])
                        - (x[quadruplet[:, 2]] - x[quadruplet[:, 3]]))
        log_likelihood = (np.log(np.maximum(prob ** answer, float_min))
                          + np.log(np.maximum((1 - prob) ** (1 - answer), float_min)))
        return log_likelihood.sum()

    def fit(self, X: utils.Query, y: np.ndarray = None) -> 'MLDS':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            init: Initial embedding for optimization
        Returns:
            This estimator
        """
        random_state = check_random_state(self.random_state)
        n_objects = X.max() + 1

        triplets, answer = utils.check_query_response(X, y, result_format='list-boolean')
        quads = triplets[:, [1, 0, 0, 2]]
        if self.method.lower() == 'glm':
            X01, rows = np.zeros((len(quads), n_objects)), np.arange(len(triplets))
            X01[rows, quads[:, 0]] += 1
            X01[rows, quads[:, 3]] += 1
            X01[rows, quads[:, 1]] -= 1
            X01[rows, quads[:, 2]] -= 1
            glm = LogisticRegression(verbose=self.verbose, max_iter=self.max_iter,
                                     fit_intercept=False, random_state=random_state)
            glm.fit(X01, answer.astype(int))
            self.embedding_ = glm.coef_.reshape(-1, 1)
            self.log_likelihood_ = glm.predict_log_proba(X01)[rows, answer.astype(int)].mean()
            self.n_iter_ = glm.n_iter_
        elif self.method.lower() == 'optim':
            def objective(*args):
                return -self._log_likelihood(*args)

            init = np.linspace(0, 1, n_objects)
            result = minimize(objective, init, args=(quads, answer),
                              method='L-BFGS-B', options=dict(maxiter=self.max_iter, disp=self.verbose))
            if self.verbose and not result.success:
                print(f"MLDS's optimization failed with reason: {result.message}.")
            self.embedding_ = result.x.reshape(-1, 1)
            self.log_likelihood_ = -result.fun
            self.n_iter_ = result.nit
        else:
            raise ValueError(f"Expects optimizer method in {{glm, optim}}, got {self.method}")

        self.embedding_ -= self.embedding_.min()
        return self
