from typing import Union, Optional, Tuple

import numpy as np
import scipy
import sparse
from sklearn.utils import check_X_y, check_array

from ._data_format import data_format, check_format
from ._data_format import QueryFormat, ResponseFormat, Format
from ._typing import Query


def _check_list_query_response(query, response):
    if response is None:
        return check_array(query, dtype=np.uint32), None
    else:
        return check_X_y(query, response, dtype=np.uint32)


def _unroll_responses(query: Optional[np.ndarray], response: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Repeat entries, whose responses are multiple of -1 or 1.
    """
    frequency = np.maximum(1, np.abs(response.astype(int)))  # max to not loose "undecided" (0) responses
    if np.all(frequency == 1):
        return query, response
    else:
        unrolled_response = np.repeat(response.clip(-1, 1), frequency, axis=0)
        if query is None:
            return query, unrolled_response
        else:
            return np.repeat(query, frequency, axis=0), unrolled_response


def _standardize_list_query(query):
    if query.shape[1] == 3:
        is_sorted = query[:, 1] <= query[:, 2]
        if np.all(is_sorted):
            return query, []
        else:
            query = np.where(np.c_[is_sorted, is_sorted, is_sorted], query, query[:, [0, 2, 1]])
            return query, ~is_sorted
    elif query.shape[1] == 4:
        is_sorted_left = query[:, 0] <= query[:, 1]
        is_sorted_right = query[:, 2] <= query[:, 3]
        is_sorted = np.argmin(query, axis=1) < 2

        if np.all(is_sorted_left & is_sorted_right & is_sorted):
            return query, []
        else:
            query = np.where(np.c_[is_sorted_left, is_sorted_left, is_sorted_right, is_sorted_right],
                             query, query[:, [1, 0, 3, 2]])
            query = np.where(np.c_[is_sorted, is_sorted, is_sorted, is_sorted], query, query[:, [2, 3, 0, 1]])
            return query, ~is_sorted
    else:
        raise ValueError("Expects triplet or quadruplet query to standardize.")


def check_bool_list_query_response(query, response, standard: bool = True):
    query, response = _check_list_query_response(query, response)
    __, input_response_format = data_format(query, response)

    if input_response_format is ResponseFormat.BOOLEAN:
        bool_response = response
    elif input_response_format is ResponseFormat.COUNT:
        if np.any(response == 0):
            raise ValueError("Undecided responses (0) cannot be represented as order or bool.")
        query, response = _unroll_responses(query, response)
        bool_response = ((response + 1) / 2).astype(bool)
    elif input_response_format is ResponseFormat.ORDER:
        bool_response = np.full((query.shape[0],), True)

    if standard:
        query, mask = _standardize_list_query(query)
        bool_response[mask] = ~bool_response[mask]

    return query, bool_response


def check_count_list_query_response(query, response, standard: bool = True):
    __, input_response_format = data_format(query, response)
    query, response = _check_list_query_response(query, response)
    if input_response_format is ResponseFormat.COUNT:
        query, count_response = _unroll_responses(query, response)
    if input_response_format is ResponseFormat.BOOLEAN:
        count_response = response.astype(int) * 2 - 1
    elif input_response_format is ResponseFormat.ORDER:
        count_response = np.full((query.shape[0],), 1)

    if standard:
        query, mask = _standardize_list_query(query)
        count_response[mask] *= -1
    return query, count_response


def check_order_list_query_response(query, response):
    query, response = _check_list_query_response(query, response)
    __, input_response_format = data_format(query, response)

    if input_response_format is ResponseFormat.COUNT:
        if np.any(response == 0):
            raise ValueError("Undecided responses (0) cannot be represented as order or bool.")
        query, response = _unroll_responses(query, response)
        filter = response == 1
    elif input_response_format is ResponseFormat.BOOLEAN:
        filter = response

    if input_response_format is not ResponseFormat.ORDER:
        if query.shape[1] == 3:
            return np.where(np.c_[filter, filter, filter], query, query[:, [0, 2, 1]])
        elif query.shape[1] == 4:
            return np.where(np.c_[filter, filter, filter, filter], query, query[:, [2, 3, 0, 1]])
        else:
            raise ValueError("Expects triplet or quadruplet query to convert.")
    else:
        return query


def check_list_query_response(query: np.ndarray,
                              response: np.ndarray,
                              result_format: Optional[Format] = None,
                              standard: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Input validation for the array-like triplet format.

        See documentation of check_triplets.
    """
    if isinstance(result_format, str) and '-' not in result_format:
        result_format = QueryFormat.LIST.value + '-' + result_format
    if isinstance(result_format, ResponseFormat):
        result_format = (QueryFormat.LIST, result_format)
    query_format, response_format = check_format(result_format, query, None)
    if query_format is not QueryFormat.LIST:
        raise ValueError(f"Expects result_format list-..., got {result_format}.")

    if response_format is ResponseFormat.ORDER:
        return check_order_list_query_response(query, response)
    elif response_format is ResponseFormat.BOOLEAN:
        return check_bool_list_query_response(query, response, standard=standard)
    elif response_format is ResponseFormat.COUNT:
        return check_count_list_query_response(query, response, standard=standard)
    else:
        raise ValueError(f"Response format {response_format.value} not supported.")


def check_tensor_query_response(query: Union[sparse.COO, scipy.sparse.spmatrix],
                                result_format: Optional[Format], standard: bool) -> sparse.COO:
    """ Input validation for the sparse matrix triplet format.

        See documentation of check_triplets.
    """
    if isinstance(result_format, str) and '-' not in result_format:
        result_format = QueryFormat.TENSOR.value + '-' + result_format
    if isinstance(result_format, ResponseFormat):
        result_format = (QueryFormat.TENSOR, result_format)
    format = check_format(result_format, query, None)
    if format[0] is not QueryFormat.TENSOR or format[1] is not ResponseFormat.COUNT:
        raise ValueError(f"Expects result_format tensor-count, got {result_format}.")

    if isinstance(query, scipy.sparse.spmatrix):
        n_objects = query.shape[0]
        n_dim = int(np.ceil(np.log(np.product(query.shape)) / np.log(n_objects)))
        query = sparse.COO.from_scipy_sparse(query)
    else:
        n_objects = max(query.shape)
        n_dim = len(query.shape)

    expected_shape = n_dim * (n_objects,)
    if len(query.shape) != n_dim or np.any(np.not_equal(query.shape, expected_shape)):
        query = query.reshape(expected_shape)

    if standard:
        not_sorted = query.coords[1, :] > query.coords[2, :]
        if np.any(not_sorted):
            new_coords = np.c_[query.coords[:, ~not_sorted], query.coords[[0, 2, 1]][:, not_sorted]]
            new_data = np.r_[query.data[~not_sorted], -query.data[not_sorted]]
            query = sparse.COO(new_coords, new_data, shape=query.shape)

    return query


def check_response(response: np.ndarray, result_format: Optional[Format] = None) -> np.ndarray:
    """ Input validation for query responses.

    Checks response shape and datatype. Converts between count and boolean.

    Args:
        response: Either array_like with index-query or sparse matrix.
        result_format: One of 'boolean', or 'count'. If none, format is not changed.

    Returns:
        response list
    """
    if isinstance(result_format, str) and '-' not in result_format:
        result_format = 'list-' + result_format
    if isinstance(result_format, ResponseFormat):
        result_format = (QueryFormat.LIST, result_format)
    result_format = check_format(result_format, [], response)
    if result_format[0] is not QueryFormat.LIST or result_format[1] is ResponseFormat.ORDER:
        raise ValueError(f"Expects result format list-boolean or list-count, got {result_format}.")

    dummy_query = np.empty_like(response).reshape(-1, 1)
    return check_list_query_response(dummy_query, response, standard=False, result_format=(result_format))[1]


def check_query(query: Query, result_format: Optional[Format] = None) -> Query:
    """ Input validation for queries.

    Checks query shape and datatype.
    Converts between array (T-STE style) and sparse matrix format for query.

    Args:
        query: Either array_like with index-query or sparse matrix.
        result_format: One of 'list', or 'tensor'. If none, format is not changed.

    Returns:
        If response_format='tensor', a three-dimensional sparse.COO matrix is returned.
        The three dimensions all have size 'n_objects'.
        The entry query[i, j, k] indicates the response on ij <= jk.
        It is -1 if wrong, 0 if undecidable, and 1 if correct.

        If response_format='list',
        a numpy array of shape (n_samples, 3) is returned.
        Each row (i, j, k) indicates, ij <= ik.

    Raises:
        ValueError: If the array_like input has the wrong shape, or response types cannot be converted.
                    This happens e.g. if undecided (0) response, should be converted to ordered or boolean response.
    """
    if isinstance(result_format, str) and '-' not in result_format:
        result_format = result_format + '-count'
    if isinstance(result_format, QueryFormat):
        result_format = (result_format, ResponseFormat.COUNT)
    result_format = check_format(result_format, query, None)[0], ResponseFormat.COUNT
    query = check_query_response(query, None, standard=True, result_format=result_format)
    if result_format[0] is QueryFormat.TENSOR:
        return sparse.elemwise(np.abs, query)
    else:
        return query[0]


def check_query_response(query: Union[Query], response: Optional[np.ndarray] = None,
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
    input_query_format, input_response_format = data_format(query, response)
    output_query_format, output_response_format = check_format(result_format, query, response)

    if output_query_format is QueryFormat.TENSOR:
        if input_query_format is QueryFormat.LIST:
            query, response = check_list_query_response(query, response, (QueryFormat.LIST, output_response_format),
                                                        standard=False)
            shape = query.shape[1] * (int(query.max() + 1),)
            query = sparse.COO(query.T, response, shape=shape)
        return check_tensor_query_response(query, (output_query_format, output_response_format), standard=standard)
    elif output_query_format is QueryFormat.LIST:
        if input_query_format is QueryFormat.TENSOR:
            query = check_tensor_query_response(query, (QueryFormat.TENSOR, input_response_format), standard=False)
            query, response = query.coords.T, query.data
        return check_list_query_response(query, response, (output_query_format, output_response_format),
                                         standard=standard)