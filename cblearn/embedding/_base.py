from typing import Optional

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from cblearn import datasets
from cblearn import utils
from cblearn import metrics


class TripletEmbeddingMixin():
    def _more_tags(self):
        return {
            'requires_positive_X': True,
            'requires_positive_y': True,
            'X_types': ['categorical'],
            'binary_only': True,  # enforce binary y in tests
            'triplets': True  # enforce triplet X in tests
        }

    def fit_transform(self, X: utils.Query, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            self.fit(X, **fit_params)
            return self.embedding_
        else:
            # fit method of arity 2 (supervised transformation)
            self.fit(X, y, **fit_params)
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
