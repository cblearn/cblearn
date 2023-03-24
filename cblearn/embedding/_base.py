from typing import Optional
from numpy.typing import ArrayLike

import numpy as np
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d

import cblearn as cbl
from cblearn import Comparison
from cblearn import datasets


class TripletEmbeddingMixin(TransformerMixin, ClassifierMixin):
    def _more_tags(self):
        return {
            'requires_positive_X': True,
            'requires_positive_y': True,
            'X_types': ['categorical'],
        }

    def _prepare_data(self, X: Comparison, y: ArrayLike, quadruplets=False, return_y=True,
                      ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Validate `X` and `y` and binarize `y`.

        Args:
            X: Training data.
            y: Target values.

        Returns:
            X: (n_samples, n_features), Validated training data.
            y: (n_samples,), Validated target values.
        """
        if y is None:
            self.classes_ = np.array([-1, 1])
        else:
            y = column_or_1d(y, warn=True)
            self.classes_, y = np.unique(y, return_inverse=True)

        if quadruplets:
            return cbl.check_quadruplets(X, y, return_y=return_y)
        else:
            return cbl.check_triplets(X, y, return_y=return_y)

    def transform(self, X: Comparison = None, y: Optional[ArrayLike] = None) -> np.ndarray:
        check_is_fitted(self, 'embedding_')
        return self.embedding_

    def decision_function(self, X: Comparison) -> np.ndarray:
        check_is_fitted(self, 'embedding_')
        X, y = datasets.triplet_response(X, self.embedding_, result_format='list-count')
        return y

    def predict(self, X: Comparison) -> np.ndarray:
        scores = self.decision_function(X)

        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = np.argmax(scores, axis=1)

        return np.take(self.classes_, indices, axis=0)

    def score(self, X: Comparison, y: ArrayLike = None, sample_weight=None) -> float:
        """ Triplet score on the estimated embedding.

        Args:
            X: Triplet or quadruplet comparisons.
            y: Binary responses {-1, 1}.
            sample_weight: Individual weights for each sample.
        Returns.
            Fraction of correct triplets.
        """
        X, y = cbl.check_triplets(X, y, return_y=True)
        return ClassifierMixin.score(self, X, y, sample_weight=None)
