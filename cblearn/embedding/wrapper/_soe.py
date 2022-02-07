from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding.wrapper._r_base import RWrapperMixin


__doctest_requires__ = {'SOE': ['rpy2']}


class SOE(BaseEstimator, TripletEmbeddingMixin, RWrapperMixin):
    """ A soft ordinal embedding estimator, wrapping an R implementation.

        The wrapped R package is the reference implementation of SOE [1]_.

        Attributes:
            embedding_: Final embedding, shape (n_objects, n_components)
            stress_: Final value of the SOE stress corresponding to the embedding.


        >>> from cblearn import datasets
        >>> import doctest; doctest.ELLIPSIS_MARKER = "-output from R-"
        >>> triplets = datasets.make_random_triplets(np.random.rand(15, 2), result_format='list-order', size=1000)
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
        >>> estimator = SOE(verbose=True).fit(triplets) # doctest: +ELLIPSIS
        -output from R-
        >>> estimator.embedding_.shape
        (15, 2)


        References
        ----------
        .. [1] Terada, Y., & Luxburg, U. (2014). Local ordinal embedding.
               International Conference on Machine Learning, 847–855.
        """

    def __init__(self, n_components=2, n_init=10, margin=.1, max_iter=1000, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None):
        """
        Args:
            n_components:
                The dimension of the embedding.
            margin:
                Scale parameter which only takes strictly positive value.
            max_iter:
                Maximum number of optimization iterations.
            verbose:
                Enable verbose output.
            random_state:
                The seed of the pseudo random number generator used to initialize the optimization.
        """
        self.n_components = n_components
        self.margin = margin
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: utils.Query, y: np.ndarray = None, init: np.ndarray = None,
            n_objects: Optional[int] = None) -> 'SOE':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            init: Initial embedding for optimization
        Returns:
            self.
        """
        loe = self.import_r_package('loe')
        random_state = check_random_state(self.random_state)
        self.seed_r(random_state)

        if self.verbose:
            report_every = 100
        else:
            import rpy2.rinterface_lib

            rpy2.rinterface_lib.callbacks.consolewrite_print = lambda prompt: None
            report_every = self.max_iter

        triplets = utils.check_query_response(X, y, result_format='list-order')
        quadruplets = triplets[:, [1, 0, 0, 2]]  # type: ignore
        quadruplets = quadruplets.astype(np.int32) + 1  # R is 1-indexed, int32

        if init is None:
            init = 'rand'
        if not n_objects:
            n_objects = len(np.unique(quadruplets))

        self.stress_ = np.infty
        soe_result = loe.SOE(CM=quadruplets, N=n_objects, p=self.n_components, c=self.margin,
                             maxit=self.max_iter, report=report_every, iniX=init,
                             rnd=quadruplets.shape[0])
        i_stress = soe_result.rx2("str")[0]
        if i_stress < self.stress_:
            self.stress_ = i_stress
            self.embedding_ = np.asarray(soe_result.rx2("X"))

        return self

    def _more_tags(self):
        return {
            **TripletEmbeddingMixin._more_tags(self),
            'Xfail': [
                'check_transformer_n_iter',  # the R package does not return n_iter
            ]
        }
