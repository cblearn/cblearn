import numbers
from typing import Optional
import warnings

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from cblearn import datasets
from cblearn import utils
from cblearn import metrics


def check_n_components(n_components, name: str = "n_components") -> int:
    """ Validate the dimensionality of an embedding.

    Args:
        n_components: The number of embedding dimensions.
        name: The parameter name to use in the error message.
    Returns:
        The validated dimensionality as a Python integer.
    Raises:
        ValueError: If n_components is not a positive integer.
    """
    if isinstance(n_components, bool) or not isinstance(n_components, numbers.Integral):
        raise ValueError(f"Expects {name} to be a positive integer, "
                         f"got {n_components!r} of type {type(n_components).__name__}.")
    if n_components < 1:
        raise ValueError(f"Expects {name} to be a positive integer, got {n_components}.")
    return int(n_components)


class TripletEmbeddingMixin(TransformerMixin):
    def __sklearn_tags__(self):
        """ Estimator tags, as used by scikit-learn's common estimator checks.

        Uses the dataclass-based tag API introduced in scikit-learn 1.6,
        which replaced the dictionary returned by the former _more_tags.
        Triplet estimators are recognized in the tests by their class
        (see TripletEmbeddingMixin), because scikit-learn's Tags dataclass
        cannot carry library specific keys.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True  # was requires_positive_X
        tags.input_tags.categorical = True  # was X_types=['categorical']
        tags.target_tags.positive_only = True  # was requires_positive_y
        tags.transformer_tags.preserves_dtype = []  # .transform does not preserve dtype
        return tags

    def _validate_n_components(self) -> int:
        """ Validate the estimator's embedding dimensionality.

        The estimators call this at the beginning of .fit, not in __init__:
        scikit-learn requires that __init__ stores its parameters unaltered
        and raises no errors (check_do_not_raise_errors_in_init_or_set_params).

        Returns:
            The validated dimensionality as a Python integer.
        Raises:
            ValueError: If self.n_components is not a positive integer.
        """
        return check_n_components(self.n_components)

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
