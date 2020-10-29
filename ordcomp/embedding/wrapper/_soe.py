from typing import Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
import numpy as np

from ordcomp import utils
from ordcomp.embedding.wrapper._r_base import RWrapper


class SOE(BaseEstimator, TransformerMixin, RWrapper):
    """ A soft ordinal embedding estimator, wrapping an R implementation.

        The wrapped R package is the reference implementation of SOE [1]_.

        Attributes:
            embedding_: Final embedding, shape (n_objects, n_components)
            stress_: Final value of the SOE stress corresponding to the embedding.

        Examples:
        >>> from ordcomp import datasets
        >>> triplets = datasets.make_random_triplets(np.random.rand(15, 2), 1000, response_type='implicit')
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
        >>> estimator = SOE(verbose=False)
        >>> embedding = estimator.fit_transform(triplets) # doctest:+ELLIPSIS
        initial  value ...
        final  value ...
        converged
        >>> embedding.shape
        (15, 2)

        References
        ----------
        .. [1] Terada, Y., & Luxburg, U. (2014). Local ordinal embedding. International Conference on Machine Learning, 847–855.
        """

    def __init__(self, n_components=2, n_init=10, C=.1, max_iter=1000, verbose=False,
                 random_state: Union[None, int, np.random.RandomState] = None):
        """
        Args:
            n_components : int, default=2
                The dimension of the embedding.
            n_init: int, default=10
                Number of times the BFGS algorithm will be run with different initializations.
                The final result will be the output of the run with the smallest final stress.
            C: float, default=.1
                Scale parameter which only takes strictly positive value.
            max_iter: int, default=1000
                Maximum number of optimization iterations.
            verbose: boolean, default=False
                Enable verbose output.
            random_state: int, RandomState instance or None, default=None
                The seed of the pseudo random number generator used to initialize the optimization.
        """
        self.n_components = n_components
        self.n_init = n_init
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

        self.import_r_package('loe')

    def _more_tags(self):
        return {
            'requires_positive_X': True,
            'requires_positive_y': True,
            'X_types': ['categorical'],
        }

    def fit(self, X, y=None, init=None, n_objects=None):
        """Computes the embedding.
        Parameters
        ----------
        X : array-like, shape (n_samples, 3 or 4)
            The training input samples.
        y : Ignored
        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)
        self.seed_r(random_state)

        if self.verbose:
            report_every = 100
        else:
            report_every = self.max_iter

        triplets = utils.check_triplets(X, y, format='array', response_type='implicit')
        quadruplets = triplets[:, [1, 0, 0, 2]].astype(np.int32) + 1  # R is 1-indexed, int32

        if not init:
            init = 'rand'
            n_init = self.n_init
        else:
            n_init = 1
        if not n_objects:
            n_objects = len(np.unique(quadruplets))

        self.stress_ = np.infty
        for i_init in range(n_init):
            soe_result = self.loe.SOE(CM=quadruplets, N=n_objects, p=self.n_components, c=self.C,
                                      maxit=self.max_iter, report=report_every, iniX=init,
                                      rnd=quadruplets.shape[0])
            i_stress = soe_result.rx2("str")[0]
            if i_stress < self.stress_:
                self.stress_ = i_stress
                self.embedding_ = np.asarray(soe_result.rx2("X"))

        return self

    def transform(self, X=None):
        """ Returns the embedded coordinates.
        Refer parameters to :meth:`~.fit`.
        Returns
        -------
        X_new : array-like, shape (n_object, 1)
            Embedding coordinates of objects.
        """
        return self.embedding_

