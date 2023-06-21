from typing import Union, Optional, Tuple

import numpy as np
from sklearn.utils import deprecated

from ._data_format import check_format
from ._data_format import QueryFormat, ResponseFormat, Format
from ._typing import Query
from cblearn import check_triplets, check_quadruplets


def _check_triplets_quadruplets(*args, **kwargs):
    try:
        return check_triplets(*args, **kwargs)
    except ValueError:
        return check_quadruplets(*args, **kwargs)


@deprecated("Use cblearn.check_triplets and the other more specialized functions instead.")
def check_query_response(query: Query, response: Optional[np.ndarray] = None,
                         result_format: Optional[Format] = None, standard: bool = True) \
        -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """ Input validation for query formats.

    Checks query-response pair for shape and datatype.
    Converts query from/to list (T-STE style) and tensor (sparse matrix) format.
    For query list, converts from/to different ordered, boolean or count responses.

    Args:
        query: Either array_like with index-query or sparse matrix.
        response: Optional response per index-triplet.
        result_format: Format of result
        standard: If true, then assures that for every triplet (i, j, k): j < k
                     This is ignored for format='list-order'.

    Returns:
        If format='tensor-count', a three-dimensional sparse.COO matrix is returned.
        The three dimensions all have size 'n_objects'.
        The entry query[i, j, k] indicates the response on ij <= jk.
        It is -1 if wrong, 0 if undecidable, and 1 if correct.

        If format='list-order',
        a numpy array of shape (n_samples, 3) is returned.
        Each row (i, j, k) indicates, ij <= ik.

        If format='list-count', two numpy arrays are of shape (n_samples, 3)
        and n_samples are returned.
        The first array contains index-query (i, j, k).
        The second array elements represent the response as described above for format='tensor-count'.

        If response_format='list-boolean', same as for response_format='list-count'.
        The responses are True/False instead of 1/-1.

    Raises:
        ValueError: If the array_like input has the wrong shape, or response format cannot be converted.
                    This happens e.g. if undecided (0) responses, should be converted to ordered or boolean responses.
    """
    output_query_format, output_response_format = check_format(result_format, query, response)

    if output_query_format is QueryFormat.TENSOR:
        return _check_triplets_quadruplets(query, response, sparse=True, canonical=standard)
    elif output_query_format is QueryFormat.LIST:
        return_y = (output_response_format is not ResponseFormat.ORDER)
        if response is not None:
            response = np.array(response)
            if response.dtype is np.dtype('bool'):
                response = response * 2 - 1
        if return_y:
            X, y = _check_triplets_quadruplets(query, response, return_y=return_y, canonical=standard)
            if output_response_format is ResponseFormat.BOOLEAN:
                y = y > 0
            return X, y
        else:
            X = _check_triplets_quadruplets(query, response, return_y=return_y, canonical=standard)
            return X