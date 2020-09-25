import enum
from typing import Union, Optional, Tuple

import numpy as np
import sparse
import scipy
from sklearn.utils import check_X_y, check_array


class ResponseType(enum.Enum):
    IMPLICIT = 'implicit'
    BOOLEAN = 'boolean'
    NUMERIC = 'numeric'


class TripletFormat(enum.Enum):
    ARRAY = 'array'
    SPARSE_MATRIX = 'spmatrix'


def _triplet_array_by_response_type(triplets: np.ndarray, responses: np.ndarray, output_response_type: ResponseType):
    if responses is None:
        input_response_type = ResponseType.IMPLICIT
    else:
        if responses.dtype == np.bool:
            input_response_type = ResponseType.BOOLEAN
        elif np.issubdtype(responses.dtype, np.number):
            input_response_type = ResponseType.NUMERIC
        else:
            raise NotImplementedError("Expects responses.dtype as numeric or boolean, got {responses.dtype}")
    if input_response_type == output_response_type:
        return triplets, responses

    if output_response_type is ResponseType.IMPLICIT:
        if input_response_type is ResponseType.NUMERIC:
            unique_responses = np.unique(responses)
            if len(unique_responses) != 2 or unique_responses[0] != -1 or unique_responses[1] != 1:
                raise ValueError(f"Expects response -1 or 1 to convert to boolean, got {unique_responses}.")
            filter = responses == 1
        elif input_response_type is ResponseType.BOOLEAN:
            filter = responses
        triplets = np.where(np.c_[filter, filter, filter], triplets, triplets[:, [0, 2, 1]])
        responses = None
    elif output_response_type is ResponseType.BOOLEAN:
        if input_response_type is ResponseType.NUMERIC:
            unique_responses = np.unique(responses)
            if len(unique_responses) != 2 or unique_responses[0] != -1 or unique_responses[1] != 1:
                raise ValueError(f"Expects response -1 or 1 to convert to boolean, got {unique_responses}.")
            responses = ((responses + 1) / 2).astype(bool)
        elif input_response_type is ResponseType.IMPLICIT:
            responses = np.full((triplets.shape[0],), True)
    elif output_response_type is ResponseType.NUMERIC:
        if input_response_type is ResponseType.BOOLEAN:
            responses = responses.astype(int) * 2 - 1
        elif input_response_type is ResponseType.IMPLICIT:
            responses = np.full((triplets.shape[0],), 1)
    return triplets, responses


def check_size(size, n_items):
    if size is None:
        return n_items
    elif isinstance(size, int) or size > 1:
        return int(size)
    elif isinstance(size, float):
        return int(size * n_items)


def check_triplet_array(triplets: np.ndarray, responses: Optional[np.ndarray] = None,
                        response_type: ResponseType = ResponseType.IMPLICIT, sort_jk: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """ Input validation for the array-like triplet format.

        See documentation of check_triplets.
    """
    if responses is None:
        triplets = check_array(triplets, dtype=np.uint)
    else:
        triplets, responses = check_X_y(triplets, responses, dtype=np.uint)
    if triplets.shape[1] != 3:
        raise ValueError(f"Expects triplet array with three columns, got shape {triplets.shape}.")
    response_type = ResponseType(response_type)

    triplets, responses = _triplet_array_by_response_type(triplets, responses, response_type)

    if sort_jk and responses is not None:
        is_sorted = triplets[:, 1] < triplets[:, 2]
        triplets = np.where(np.c_[is_sorted, is_sorted, is_sorted], triplets, triplets[:, [0, 2, 1]])
        if response_type is ResponseType.NUMERIC:
            responses[~is_sorted] *= -1
        elif response_type is ResponseType.BOOLEAN:
            responses[~is_sorted] = ~responses[~is_sorted]

    return triplets, responses


def check_triplet_spmatrix(triplets: Union[sparse.COO, scipy.sparse.spmatrix], n_objects: Optional[int] = None) -> sparse.COO:
    """ Input validation for the sparse matrix triplet format.

        See documentation of check_triplets.
    """
    if isinstance(triplets, scipy.sparse.spmatrix):
        triplets = sparse.COO.from_scipy_sparse(triplets)

    if n_objects is None:
        n_objects = int(np.ceil(np.cbrt(np.product(triplets.shape))))
    expected_shape = (n_objects, n_objects, n_objects)
    if len(triplets.shape) != 3 or np.any(np.not_equal(triplets.shape, expected_shape)):
        triplets = triplets.reshape(expected_shape)

    return triplets


def check_triplets(triplets: Union[np.ndarray, sparse.COO, scipy.sparse.spmatrix],
                   responses: Optional[np.ndarray] = None, format: TripletFormat = None,
                   response_type: ResponseType = None, n_objects: Optional[int] = None
                   ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """ Input validation for triplet formats.

    Checks triplets and responses for shape and datatype.
    Converts between array (T-STE style) and sparse matrix format for triplets.
    For array format, also converts between different response formats.

    Args:
        triplets: Either array_like with index-triplets or sparse matrix.
        responses: Optional responses per index-triplet.
        format: Either 'array' or 'spmatrix' indicating the triplet format.
        response_type: One of 'implicit', 'boolean', or 'numberic'.
                       Only required with format='array'.
        n_objects: The number of individual objects in triplets, optional.
                   If not provided with format='spmatrix', value is inferred
                   by the cube-root the shape product.

    Returns:
        If format='spmatrix', a three-dimensional sparse.COO matrix is returned.
        The three dimensions all have size 'n_objects'.
        The entry triplets[i, j, k] indicates the response on ij <= jk.
        It is -1 if wrong, 0 if undecidable, and 1 if correct.

        If format='array' and response_type='implicit',
        a numpy array of shape (n_samples, 3) is returned.
        Each row (i, j, k) indicates, ij <= ik.

        If format='array' and response_type='numeric',
        two numpy arrays are of shape (n_samples, 3) and n_samples are returned.
        The first array contains index-triplets (i, j, k).
        The second array elements represent the response as described above for format='spmatrix'.

        If format='array' and response_type='boolean', same as for response_type='numeric'.
        The responses are True/False instead of 1/-1.

    Raises:
        ValueError: If the array_like input has the wrong shape, or response types cannot be converted.
                    This happens e.g. if undecided (0) responses, should be converted to implicit or boolean responses.
    """
    format = TripletFormat(format)
    if isinstance(triplets, (scipy.sparse.spmatrix, sparse.SparseArray)):
        triplets = check_triplet_spmatrix(triplets, n_objects=n_objects)
        if format is TripletFormat.ARRAY:
            triplets, responses = check_triplet_array(triplets.coords, triplets.data, response_type=response_type)
    else:
        if format is TripletFormat.SPARSE_MATRIX:
            triplets, responses = check_triplet_array(triplets, responses, response_type=ResponseType.NUMERIC)
            triplets = sparse.COO(triplets, responses)
            responses = None
        else:
            triplets, responses = check_triplet_array(triplets, responses, response_type=response_type)

    if responses is None:
        return triplets
    else:
        return triplets, responses


def check_size(size: Union[None, int, float], max_objects: int) -> int:
    """ Convert size argument to the number of objects.

    Args:
        size: The ommited, relative, or absolute number of objects.
        max_objects: The maximum size.

    Returns:
        The absolute size, corresponding to
            max_objects, if size is None
            size, if size is int
            size * max_objects, if size is float

    Raises
       ValueError:
           If size is int and < 0 or > max_objects
           If size is float and < 0 or > 1.
    """
    if size is None:
        return max_objects
    elif isinstance(size, int) or size > 1:
        if size < 0 or size > max_objects:
            raise ValueError(f'Expects size within 0 and {max_objects}, got {size}.')
        return int(size)
    elif isinstance(size, float):
        if size < 0 or size > 1:
            raise ValueError(f'Expects size within 0 and 1, got {size}.')
        return int(size * max_objects)
