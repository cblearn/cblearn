from typing import Union

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import numpy as np

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding.wrapper._r_base import RWrapperMixin


__doctest_requires__ = {'MLDS': ['rpy2']}


class MLDS(BaseEstimator, TripletEmbeddingMixin, RWrapperMixin):
    """ A maximum-likelihood difference scaling (MLDS) estimator, wrapping the R implementation.

    Note::
        This method assumes, that the objects can be embedded in a one-dimensional space
        and that the object indices are consistent to their order in this space.

    This estimator required the R programming language
    and the R package `MLDS <https://cran.r-project.org/web/packages/MLDS/index.html>`_.
    This R package is the reference implementation of MLDS [1]_.

    Attributes:
        embedding_: array-likeThe final embedding, shape (n_objects, 1)
        log_likelihood_: The final log-likelihood of the embedding.


    >>> from cblearn import datasets
    >>> import doctest; doctest.ELLIPSIS_MARKER = "-output from R-"
    >>> triplets = datasets.make_random_triplets(np.arange(15).reshape(-1, 1), size=400, result_format='list-order')
    >>> triplets.shape, np.unique(triplets).shape
    ((400, 3), (15,))
    >>> estimator = MLDS().fit(triplets); print("...finished fit") # doctest: +ELLIPSIS
    -output from R-...finished fit
    >>> estimator.embedding_.shape
    (15, 1)


    References
    ----------
    .. [1] M Knoblauch, K., & Maloney, L. T. (2012). Modeling Psychophysical Data in R.
           Springer New York. https://doi.org/10.1007/978-1-4614-4475-6
    """

    def __init__(self, n_components: int = 1, random_state: Union[None, int, np.random.RandomState] = None, method='glm'):
        """
        Args:
            n_components: Embedding dimension for api compatibility. Only 1 is supported for MLDS.
            random_state: The seed of the pseudo random number generator used to initialize the optimization.
            method: Optimizer method, either 'glm' or 'optim'.
        """
        if n_components != 1:
            raise ValueError(f"MLDS expects n_components=1, got {n_components}")
        self.n_components = n_components
        self.random_state = random_state
        self.method = method

    def fit(self, X: utils.Query, y: np.ndarray = None) -> 'MLDS':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            init: Initial embedding for optimization
        Returns:
            This estimator
        """
        mlds = self.import_r_package('MLDS', robject_translations={"logLik.mlds": "logLik_mlds"})
        random_state = check_random_state(self.random_state)
        self.seed_r(random_state)

        triplets, answer = utils.check_query_response(X, y, result_format='list-boolean')
        triplets = triplets.astype(np.int32) + 1
        r_df = self.robjects.vectors.DataFrame({
            'resp': answer,
            's1': triplets[:, 1],
            's2': triplets[:, 0],
            's3': triplets[:, 2],
        })

        self.r_estimator_ = mlds.mlds(r_df, method=self.method)
        self.log_likelihood_ = mlds.logLik_mlds(self.r_estimator_)[0]
        self.embedding_ = np.asarray(self.r_estimator_.rx2("pscale")).reshape(-1, 1)

        return self
