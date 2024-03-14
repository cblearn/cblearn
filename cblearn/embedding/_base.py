from typing import Optional
import warnings

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from cblearn import datasets
from cblearn import utils
from cblearn import metrics


class TripletEmbeddingMixin(TransformerMixin):
    def _more_tags(self):
        return {
            'requires_positive_X': True,
            'requires_positive_y': True,
            'X_types': ['categorical'],
            'preserves_dtype': [],  # .transform does not preserve dtype
            'binary_only': True,  # enforce binary y in tests
            'triplets': True  # enforce triplet X in tests
        }

    def transform(self, X: Optional[utils.Query]):
        """ Transform the input data into the learned embedding.

        The input data can be none or an array with all or a subset of the
        triplets provided by .fit method.
        Actually, the input data is not used in this method, but is required
        for compatibility with the scikit-learn API.

        Args:
            X: Triplet answers, identical to the input in the .fit method or None.
        Returns:
            The learned embedding.
        Warns:
            If X is not the same instance as in the .fit method.
        """
        check_is_fitted(self, ['embedding_', 'fit_X_'])

        if X is not None:
            # Check if the input is a valid query, required by sklearn estimator tests
            X_order = utils.check_query(X, result_format='list-order')
            not_isin = ~utils.isin_query(X_order, self.fit_X_)
            if not_isin.any():
                # X has to be allowed for the sklearn Pipeline API.
                # https://github.com/scikit-learn/scikit-learn/blob/
                # 19f41496868a98d8326a20e2a3039b2a4e24280e/sklearn/pipeline.py#L258
                # https://github.com/scikit-learn/scikit-learn/blob/19f41496868a98d8326a20e2a3039b2a4e24280e/
                # sklearn/pipeline.py#L1302C1-L1303C85
                warnings.warn(UserWarning(
                    "Expects the same X queries in .fit and .transform (or None),"
                    f"got {X_order[not_isin]} not in fit(X).\n"
                    "Note: X can be passed for compatibility with the scikit-learn API."))

        return self.embedding_

    def predict(self, X: utils.Query, result_format: Optional[utils.Format] = None) -> np.ndarray:
        check_is_fitted(self, 'embedding_')
        result = datasets.triplet_response(X, self.embedding_, result_format=result_format)
        if isinstance(result, tuple):
            return result[1]
        else:
            return result

    def score(self, X: utils.Query, y: Optional[np.ndarray] = None) -> float:
        """ Triplet score on the estimated embedding.

        Args:
            X: Triplet answers
        Returns.
            Fraction of correct triplets.
        """
        X, y = utils.check_query_response(X, y, result_format='list-count')
        return metrics.query_accuracy(self.predict(X, result_format='list-count'), y)
