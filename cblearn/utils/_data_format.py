import enum
from typing import Union, Optional, Tuple, Sequence

import scipy
import sparse
import numpy as np

from ._typing import Query, Response


class QueryFormat(enum.Enum):
    LIST = 'list'
    TENSOR = 'tensor'


class ResponseFormat(enum.Enum):
    ORDER = 'order'
    BOOLEAN = 'boolean'
    COUNT = 'count'


Format = Union[str, Tuple[QueryFormat, ResponseFormat]]


def check_format(format: Optional[Format], default_query: Union[Query, Response],
                 default_response: Optional[np.ndarray]) -> Tuple[QueryFormat, ResponseFormat]:
    """ Validate comparison format description.

    The format of comparison data is specified as a single string separated by '-', or a tuple of format identifiers.
    Valid formats include 'list-order', 'list-boolean', 'list-count', and 'tensor-count'.
    If the format is not explicitly given, the format used by the default_triplets and default_response is returned.

    Args:
        format: Comparison format string 'query-response' or tuple of format strings (query, response).
        default_query: Comparison query to extract format from if not passed explicitly.
        default_response: Comparison response to extract format from if not passed explicitly.
    Returns:
        Tuple of format identifiers (query, response)
    Raises:
        ValueError: Format identifier is unknown.
        IndexError: Less than 2 format components.
    """
    if format is None:
        return data_format(default_query, default_response)
    else:
        if isinstance(format, str):
            format_parts = format.split('-')
            return QueryFormat(format_parts[0]), ResponseFormat(format_parts[1])
        elif isinstance(format, tuple):
            return QueryFormat(format[0]), ResponseFormat(format[1])
        else:
            raise ValueError(f"Expects either format as string, enum-tuple or None; got {format}.")


def data_format(query: Union[Query], response: Optional[np.ndarray] = None
                ) -> Tuple[QueryFormat, ResponseFormat]:
    """ Extract format of comparison data.

    Args:
        query: Comparison query to extract format.
        response: Comparison response to extract format.
    Returns:
        Tuple of format identifiers (query, response)
    Raises:
        TypeError: Invalid type of data.
    """
    if isinstance(query, (scipy.sparse.spmatrix, sparse.SparseArray)):
        query_format = QueryFormat.TENSOR
    elif isinstance(query, (Sequence, np.ndarray)):
        query_format = QueryFormat.LIST
    elif query is None:
        query_format = None
    else:
        raise ValueError(f"Expects query as sequence, array, or sparse array; got {query}")

    if response is None:
        if query_format is QueryFormat.TENSOR:
            response_dtype = query.dtype
        elif query_format is QueryFormat.LIST:
            return query_format, ResponseFormat.ORDER
        else:
            return query_format, None
    elif isinstance(response, (Sequence, np.ndarray)):
        response_dtype = np.asarray(response).dtype
    else:
        return query_format, None

    if response_dtype == bool:
        return query_format, ResponseFormat.BOOLEAN
    elif np.issubdtype(response_dtype, np.number):
        return query_format, ResponseFormat.COUNT
    else:
        raise ValueError(f"Expects response dtype bool or numeric, got {response_dtype}")