from typing import Dict, Union, List, Tuple, Optional

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import numpy as np


def _unique_firstaxis(X, return_inverse: bool = False):
    """ Find sorted, unique array elements.

        Has a (slow) fallback, if the numpy array is mixed typed and
        cannot be used with numpy's unique method.

    >>> _unique_firstaxis([[0.1, 'high'], [0.3, 'low'], [0.1, 'high'], [0.4, 'low']]).tolist()
    [['0.1', 'high'], ['0.3', 'low'], ['0.4', 'low']]
    >>> u, i = _unique_firstaxis([[0.1, 'high'], [0.3, 'low'], [0.1, 'high'], [0.4, 'low']], return_inverse=True)
    >>> u.tolist(), i.tolist()
    ([['0.1', 'high'], ['0.3', 'low'], ['0.4', 'low']], [0, 1, 0, 2])
    """
    X = check_array(X, dtype=None, ensure_2d=True)
    if X.dtype == object:
        py_X = tuple(map(tuple, X))
        uniques = sorted(set(py_X))
        if return_inverse:
            ix_map = {val: ix for ix, val in enumerate(uniques)}
            indices = [ix_map[val] for val in py_X]
            return np.array(uniques, dtype=X.dtype), np.array(indices)
        else:
            return uniques
    else:
        return np.unique(X, axis=0, return_inverse=return_inverse)


class MultiColumnLabelEncoder(LabelEncoder):
    """ Encoder for objects that are a combination of labels in multiple columns.

    Extends the function of scikit-learn's label encoder to 2d arrays.
    See :class:`sklearn.preprocessing.LabelEncoder` for more information.

    >>> encoder = MultiColumnLabelEncoder()
    >>> label_data = [[0.1, 'high'], [0.3, 'low'], [0.1, 'high'], [0.1, 'low']]
    >>> encoder.fit(label_data).transform(label_data).tolist()
    [0, 2, 0, 1]
    >>> encoder.fit_transform(label_data).tolist()
    [0, 2, 0, 1]
    >>> encoder.inverse_transform([2, 1, 0]).tolist()
    [['0.3', 'low'], ['0.1', 'low'], ['0.1', 'high']]
    """
    def fit(self, X, y=None):
        self.classes_ = _unique_firstaxis(X)
        return self

    def fit_transform(self, X, y=None):
        self.classes_, indices = _unique_firstaxis(X, return_inverse=True)
        return indices

    def transform(self, X, y=None):
        # This method is a modified copy of scikit-learn's implementation
        # of sklearn.preprocessing.LabelEncoder.transform (3-clause BSC licensed).
        check_is_fitted(self)
        X = check_array(X, dtype=None)

        if len(X) == 0:
            return np.array([])

        ix = np.empty(len(X), dtype=int)
        for i, c in enumerate(self.classes_):
            c_ix = np.where((X == c).all(axis=1))[0]
            ix[c_ix] = i
        return ix

    def inverse_transform(self, X, y=None):
        return LabelEncoder.inverse_transform(self, X)


class SharedColumnEncoder(TransformerMixin, BaseEstimator):
    """ Wrapper to share an encoder across all columns.

    >>> encoder = SharedColumnEncoder(LabelEncoder())
    >>> label_data = [[0.1, 0.3, 0.4], [0.4, 0.1, 0.3], [0.5, 0.3, 0.3]]
    >>> encoder.fit(label_data).transform(label_data).tolist()
    [[0, 1, 2], [2, 0, 1], [3, 1, 1]]
    >>> encoder.fit_transform(label_data).tolist()
    [[0, 1, 2], [2, 0, 1], [3, 1, 1]]
    >>> encoder.inverse_transform([[2, 2], [1, 0], [0, 1]]).tolist()
    [[0.4, 0.4], [0.3, 0.1], [0.1, 0.3]]
    """
    def __init__(self, encoder):
        self.encoder_ = encoder

    def fit(self, X, y=None):
        X = check_array(X, allow_nd=True, dtype=None)
        self.encoder_.fit(X.reshape(-1, *X.shape[2:]))
        return self

    def fit_transform(self, X, y=None):
        X = check_array(X, allow_nd=True, dtype=None)
        long_X = self.encoder_.fit_transform(X.reshape(-1, *X.shape[2:]))
        return long_X.reshape(X.shape[:2])

    def transform(self, X, y=None):
        X = check_array(X, allow_nd=True, dtype=None)
        return self.encoder_.transform(X.reshape(-1, *X.shape[2:])).reshape(X.shape[:2])

    def inverse_transform(self, X, y=None):
        X = check_array(X, allow_nd=True, dtype=None)
        return self.encoder_.inverse_transform(X.reshape(-1, *X.shape[2:])).reshape(X.shape[0], -1)

    def _more_tags(self):
        return {}


def query_from_columns(data: Union[np.ndarray, "pandas.DataFrame"],  # noqa: F821  ignore pandas, not a library dep
                       query_columns: Union[List[str], List[int]],
                       response_columns: Optional[Union[List[str], List[int], str, int]] = None,
                       response_map: Optional[Dict[str, Union[bool, int]]] = None,
                       return_transformer: bool = False) \
        -> Union[Tuple[np.ndarray, np.ndarray],
                 Tuple[Tuple[np.ndarray, np.ndarray], Tuple[TransformerMixin, TransformerMixin]]]:
    """ Extract queries from objects in columns or dataframes.

        The objects in the column data might be defined by a single or multiple numerical attributes.
        Then this function assigns to each object an index and returns query and response based on object indices,
        as required by most library functions.
        If attributes are non-numeric, consider first encoding them with :class:`sklearn.preprocessing.LabelEncoder`.

        >>> import pandas as pd
        >>> frame = pd.DataFrame({'alpha1': [0.1, 0.7, 0.1], 'tau1': [0, 0, 1],
        ...                       'alpha2': [0.3, 0.3, 0.7], 'tau2': [1, 0, 0],
        ...                       'alpha3': [0.7, 0.3, 0.7], 'tau3': [0, 1, 0], 'Response': [1, 0, 0]})
        >>> q, r = query_from_columns(frame, ['alpha1', 'alpha2', 'alpha3'], 'Response', response_map={1: True, 0: False})
        >>> q.tolist(), r.tolist()
        ([[0, 1, 2], [2, 1, 1], [0, 2, 2]], [True, False, False])
        >>> q, r = query_from_columns(np.array(frame), [0, 2, 4], response_columns=-1, response_map={1: True, 0: False})
        >>> q.tolist(), r.tolist()
        ([[0, 1, 2], [2, 1, 1], [0, 2, 2]], [True, False, False])
        >>> q, r = query_from_columns(frame, [('alpha1', 'tau1'), ('alpha2', 'tau2'), ('alpha3', 'tau3')],
        ...                           response_columns='Response', response_map={1: True, 0: False})
        >>> q.tolist(), r.tolist()
        ([[0, 3, 4], [4, 2, 3], [1, 4, 4]], [True, False, False])

        The transformers can be used to get object attributes from the object index.

        >>> (q,r), (q_transform, r_transform) = query_from_columns(
        ...     np.array(frame), [0, 2, 4], -1, {1: True, 0: False}, return_transformer=True)
        >>> q_transform.inverse_transform(q).tolist()
        [[0.1, 0.3, 0.7], [0.7, 0.3, 0.3], [0.1, 0.7, 0.7]]


        Args:
             data: Tabular query representation (n_queries, n_columns)
             query_columns: Indices or column-labels in data per query entry.
                            Columns can be grouped as tuples, if multiple columns define an object.
             response_columns: Indices or column-labels in data per response entry.
             response_map: Dictionary mapping the response entries in data to {-1, 1} or {False, True}.
             return_transformer: If true, transformer objects for the query and response are returned.
        Returns:
            Tuple with arrays for the queries and responses.

            If return_transform=True, an additional tuple with transformer objects is returned.

    """
    if not hasattr(data, 'columns'):  # is no pandas Dataframe?
        data = check_array(data, dtype=None).T
    query_data = np.swapaxes(np.stack([data[np.array(c)] for c in query_columns]), 0, 1)
    if len(query_data.shape) == 3:
        query_enc = SharedColumnEncoder(MultiColumnLabelEncoder())
    else:
        query_enc = SharedColumnEncoder(LabelEncoder())
    query = query_enc.fit_transform(query_data)

    if response_columns:
        inverse_map = {v: k for k, v in response_map.items()}
        response_enc = FunctionTransformer(func=np.vectorize(response_map.get),
                                           inverse_func=np.vectorize(inverse_map.get), check_inverse=False)
        response = response_enc.fit_transform(data[response_columns])
        if return_transformer:
            return (query, response), (query_enc, response_enc)
        else:
            return query, response
    else:
        if return_transformer:
            return query, query_enc
        else:
            return query
