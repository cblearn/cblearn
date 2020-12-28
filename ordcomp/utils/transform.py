import enum
from typing import Union, Optional, Tuple, Sequence

import numpy as np
import scipy
import sparse
from sklearn.utils import check_X_y, check_array


_SPARSE_TYPES = (scipy.sparse.spmatrix, sparse.SparseArray)
Triplets = Union[np.ndarray, sparse.COO, scipy.sparse.spmatrix]
TripletAnswers = Union[Triplets, Tuple[Triplets, np.ndarray]]


class QuestionFormat(enum.Enum):
    LIST = 'list'
    TENSOR = 'tensor'


class AnswerFormat(enum.Enum):
    ORDER = 'order'
    BOOLEAN = 'boolean'
    COUNT = 'count'


def triplet_format(triplets: Union[Triplets, TripletAnswers], answers: Optional[np.ndarray] = None
                   ) -> Tuple[QuestionFormat, AnswerFormat]:
    if isinstance(triplets, tuple) and answers is None:
        triplets, answers = triplets

    if answers is None:
        if isinstance(triplets, _SPARSE_TYPES):
            return QuestionFormat.TENSOR, AnswerFormat.COUNT
        elif isinstance(triplets, (Sequence, np.ndarray)):
            return QuestionFormat.LIST, AnswerFormat.ORDER
    elif isinstance(answers, (Sequence, np.ndarray)):
        answer_type = np.asarray(answers).dtype
        if answer_type == np.bool:
            return QuestionFormat.LIST, AnswerFormat.BOOLEAN
        elif np.issubdtype(answer_type, np.number):
            return QuestionFormat.LIST, AnswerFormat.COUNT
        else:
            raise TypeError(f"Unknown triplet question/answer format for answer.dtype=={answer_type}.")
    raise TypeError(f"Unknown triplet question/answer format for {type(triplets)}/{type(answers)}.")


def _triplet_array_by_answer_format(triplets: np.ndarray, answers: np.ndarray,
                                    input_answer_format: AnswerFormat, output_answer_format: AnswerFormat):
    # quick exit if no conversion is necessary
    if input_answer_format == output_answer_format:
        return triplets, answers

    if input_answer_format is AnswerFormat.COUNT and np.any(answers == 0):
        raise ValueError("Undecided triplet answers (0) cannot be converted to ordered or boolean.")

    if output_answer_format is AnswerFormat.ORDER:
        if input_answer_format is AnswerFormat.COUNT:
            filter = answers == 1
        elif input_answer_format is AnswerFormat.BOOLEAN:
            filter = answers
        triplets = np.where(np.c_[filter, filter, filter], triplets, triplets[:, [0, 2, 1]])
        answers = None
    elif output_answer_format is AnswerFormat.BOOLEAN:
        if input_answer_format is AnswerFormat.COUNT:
            answers = ((answers + 1) / 2).astype(bool)
        elif input_answer_format is AnswerFormat.ORDER:
            answers = np.full((triplets.shape[0],), True)
    elif output_answer_format is AnswerFormat.COUNT:
        if input_answer_format is AnswerFormat.BOOLEAN:
            answers = answers.astype(int) * 2 - 1
        elif input_answer_format is AnswerFormat.ORDER:
            answers = np.full((triplets.shape[0],), 1)

    return triplets, answers


def _check_triplet_array(triplets: np.ndarray,
                         answers: np.ndarray,
                         answer_format: AnswerFormat,
                         sort_others: bool) -> Tuple[np.ndarray, np.ndarray]:
    """ Input validation for the array-like triplet format.

        See documentation of check_triplets.
    """
    __, input_answer_format = triplet_format(triplets, answers)
    if answers is None:
        triplets = check_array(triplets, dtype=np.uint32)
    else:
        triplets, answers = check_X_y(triplets, answers, dtype=np.uint32)

    if triplets.shape[1] != 3:
        raise ValueError(f"Expects triplet array with three columns, got shape {triplets.shape}.")

    # repeat triplets with answers >1/<-1
    if input_answer_format is AnswerFormat.COUNT and np.any(np.abs(answers) > 1):
        answer_frequency = np.maximum(1, np.abs(answers.data))
        triplets, answers = (np.repeat(triplets, answer_frequency, axis=0),
                             np.repeat(answers.clip(-1, 1), answer_frequency, axis=0))

    triplets, answers = _triplet_array_by_answer_format(triplets, answers, input_answer_format, answer_format)

    if sort_others and answers is not None:
        is_sorted = triplets[:, 1] <= triplets[:, 2]
        triplets = np.where(np.c_[is_sorted, is_sorted, is_sorted], triplets, triplets[:, [0, 2, 1]])
        if answer_format is AnswerFormat.COUNT:
            answers[~is_sorted] *= -1
        elif answer_format is AnswerFormat.BOOLEAN:
            answers[~is_sorted] = ~answers[~is_sorted]

    if answers is None:
        return triplets
    else:
        return triplets, answers


def _check_triplet_spmatrix(triplets: Union[sparse.COO, scipy.sparse.spmatrix],
                            answer_format: AnswerFormat, sort_others: bool, n_objects: Optional[int]) -> sparse.COO:
    """ Input validation for the sparse matrix triplet format.

        See documentation of check_triplets.
    """
    if answer_format is not AnswerFormat.COUNT:
        raise NotImplementedError("The sparse tensor triplet format is implemented for count answer format,"
                                  f" got {answer_format}.")

    if isinstance(triplets, scipy.sparse.spmatrix):
        triplets = sparse.COO.from_scipy_sparse(triplets)

    if n_objects is None:
        n_objects = int(np.cbrt(np.product(triplets.shape)))
    expected_shape = (n_objects, n_objects, n_objects)
    if len(triplets.shape) != 3 or np.any(np.not_equal(triplets.shape, expected_shape)):
        triplets = triplets.reshape(expected_shape)

    if sort_others:
        not_sorted = triplets.coords[1, :] > triplets.coords[2, :]
        if np.any(not_sorted):
            new_coords = np.c_[triplets.coords[:, ~not_sorted], triplets.coords[[0, 2, 1]][:, not_sorted]]
            new_data = np.r_[triplets.data[~not_sorted], -triplets.data[not_sorted]]
            triplets = sparse.COO(new_coords, new_data)

    return triplets


def check_triplet_questions(triplets: Triplets,
                            format: Union[str, QuestionFormat, None] = None, n_objects: Optional[int] = None
                            ) -> Triplets:
    """ Input validation for triplet formats.

    Checks triplets and answers for shape and datatype.
    Converts between array (T-STE style) and sparse matrix format for triplets.
    For array format, also converts between different answer formats.

    Args:
        triplets: Either array_like with index-triplets or sparse matrix.
        answers: Optional answers per index-triplet.
        format: One of 'list', or 'tensor'. If none, format is not changed.
        sort_others: If true, then assures that for every triplet (i, j, k): j < k
                     This is ignored for answer_format='order'.
        n_objects: The number of individual objects in triplets, optional.
                   If not provided with format='sparse', value is inferred
                   by the cube-root the shape product.

    Returns:
        If answer_format='tensor', a three-dimensional sparse.COO matrix is returned.
        The three dimensions all have size 'n_objects'.
        The entry triplets[i, j, k] indicates the answer on ij <= jk.
        It is -1 if wrong, 0 if undecidable, and 1 if correct.

        If answer_format='list',
        a numpy array of shape (n_samples, 3) is returned.
        Each row (i, j, k) indicates, ij <= ik.

    Raises:
        ValueError: If the array_like input has the wrong shape, or answer types cannot be converted.
                    This happens e.g. if undecided (0) answers, should be converted to ordered or boolean answers.
    """
    input_format, __ = triplet_format(triplets)
    output_format = QuestionFormat(format or input_format)
    if output_format is QuestionFormat.TENSOR:
        if input_format is QuestionFormat.LIST:
            triplets = np.asarray(triplets)
            n_objects = n_objects or int(triplets.max() + 1)
            triplets = sparse.COO(triplets.T, np.ones(len(triplets)), shape=(n_objects, n_objects, n_objects))
        return _check_triplet_spmatrix(triplets, AnswerFormat.COUNT, sort_others=True, n_objects=n_objects)
    elif output_format is QuestionFormat.LIST:
        if input_format is QuestionFormat.TENSOR:
            triplets, answers = triplets.coords.T, triplets.data
        else:
            answers = None
        return _check_triplet_array(triplets, answers, AnswerFormat.ORDER, sort_others=True)


def check_triplet_answers(triplet_answers: Union[Triplets, TripletAnswers], answers: Optional[np.ndarray] = None,
                          question_format: Optional[Union[str, QuestionFormat]] = None,
                          answer_format: Optional[Union[str, AnswerFormat]] = None,
                          sort_others: bool = True, n_objects: Optional[int] = None
                          ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """ Input validation for triplet formats.

    Checks triplets and answers for shape and datatype.
    Converts between array (T-STE style) and sparse matrix format for triplets.
    For array format, also converts between different answer formats.

    Args:
        triplets: Either array_like with index-triplets or sparse matrix.
        answers: Optional answers per index-triplet.
        answer_format: One of 'order', 'boolean', 'numberic', or 'sparse'.
        sort_others: If true, then assures that for every triplet (i, j, k): j < k
                     This is ignored for answer_format='order'.
        n_objects: The number of individual objects in triplets, optional.
                   If not provided with format='sparse', value is inferred
                   by the cube-root the shape product.

    Returns:
        If answer_format='sparse', a three-dimensional sparse.COO matrix is returned.
        The three dimensions all have size 'n_objects'.
        The entry triplets[i, j, k] indicates the answer on ij <= jk.
        It is -1 if wrong, 0 if undecidable, and 1 if correct.

        If answer_format='order',
        a numpy array of shape (n_samples, 3) is returned.
        Each row (i, j, k) indicates, ij <= ik.

        If answer_format='count', two numpy arrays are of shape (n_samples, 3)
        and n_samples are returned.
        The first array contains index-triplets (i, j, k).
        The second array elements represent the answer as described above for format='sparse'.

        If answer_format='boolean', same as for answer_format='count'.
        The answers are True/False instead of 1/-1.

    Raises:
        ValueError: If the array_like input has the wrong shape, or answer types cannot be converted.
                    This happens e.g. if undecided (0) answers, should be converted to ordered or boolean answers.
    """
    if isinstance(triplet_answers, tuple) and answers is None:
        triplets, answers = triplet_answers
    else:
        triplets = triplet_answers
    input_question_format, input_answer_format = triplet_format(triplets, answers)
    output_answer_format = AnswerFormat(answer_format or input_answer_format)
    output_question_format = QuestionFormat(question_format or input_question_format)

    if output_question_format is QuestionFormat.TENSOR:
        if input_question_format is QuestionFormat.LIST:
            triplets, answers = _check_triplet_array(triplets, answers, sort_others=False,
                                                     answer_format=AnswerFormat.COUNT)
            n_objects = n_objects or int(triplets.max() + 1)
            triplets = sparse.COO(triplets.T, np.ones(len(triplets)), shape=(n_objects, n_objects, n_objects))
        return _check_triplet_spmatrix(triplets, output_answer_format, sort_others=sort_others, n_objects=n_objects)
    elif output_question_format is QuestionFormat.LIST:
        if input_question_format is QuestionFormat.TENSOR:
            triplets = _check_triplet_spmatrix(triplets, input_answer_format, sort_others=False, n_objects=n_objects)
            triplets, answers = triplets.coords.T, triplets.data
        return _check_triplet_array(triplets, answers, sort_others=sort_others,
                                    answer_format=output_answer_format)


def check_size(size: Union[None, int, float], max_objects: int) -> int:
    """ Convert size argument to the number of objects.

    Args:
        size: The ommited, relative, or absolute number of objects.
        max_objects: The maximum number of objects for relative size.

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
    elif size < 0:
        raise ValueError(f'Expects size above 0, got {size}.')
    elif isinstance(size, int) or size > 1:
        return int(size)
    elif isinstance(size, float):
        return int(size * max_objects)
