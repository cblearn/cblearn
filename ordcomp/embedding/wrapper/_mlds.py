from typing import Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
import numpy as np

from ... import utils
from ._r_base import RWrapper


class MLDS(BaseEstimator, TransformerMixin, RWrapper):
    """ A maximum-likelihood difference scaling (MLDS) estimator, wrapping the R implementation.

    Requires the R programming language and installed R package `MLDS <https://cran.r-project.org/web/packages/MLDS/index.html>`_.
    The wrapped R package is the reference implementation of MLDS [1]_ using generalized linear models.

    Attributes:
        embedding_: array-likeThe final embedding, shape (n_objects, 1)
        log_likelihood_: The final log-likelihood of the embedding.

    >>> from ordcomp import datasets
    >>> triplets = datasets.make_random_triplets(np.arange(15).reshape(-1, 1), 400, response_type='implicit')
    >>> triplets.shape, np.unique(triplets).shape
    ((400, 3), (15,))
    >>> estimator = MLDS()
    >>> embedding = estimator.fit_transform(triplets)
    >>> embedding.shape
    (15, 1)

    References
    ----------
    .. [1] M Knoblauch, K., & Maloney, L. T. (2012). Modeling Psychophysical Data in R. Springer New York. https://doi.org/10.1007/978-1-4614-4475-6
    """

    def __init__(self, random_state: Union[None, int, np.random.RandomState] = None):
        """
        Args:
            random_state: The seed of the pseudo random number generator used to initialize the optimization.
        """
        self.random_state = random_state
        super(RWrapper, self).__init__()
        self.import_r_package('MLDS', robject_translations={"logLik.mlds": "logLik_mlds"})

    def _more_tags(self):
        return {
            'requires_positive_X': True,
            'requires_positive_y': True,
            'X_types': ['categorical'],
        }

    def fit(self, X, y=None):
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

        triplets, response = utils.check_triplets(X, y, format='array', response_type='boolean')
        triplets = triplets.astype(np.int32) + 1
        r_df = self.robjects.vectors.DataFrame({
            'resp': response,
            's1': triplets[:, 1],
            's2': triplets[:, 0],
            's3': triplets[:, 2],
        })

        self.r_estimator_ = self.MLDS.mlds(r_df, method='glm')
        self.log_likelihood_ = self.MLDS.logLik_mlds(self.r_estimator_)[0]
        self.embedding_ = np.asarray(self.r_estimator_.rx2("pscale")).reshape(-1, 1)

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

