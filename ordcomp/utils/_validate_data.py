from typing import Union, Optional, Tuple

import numpy as np
import scipy
import sparse
from sklearn.utils import check_X_y, check_array

from ._data_format import data_format, check_format
from ._data_format import QuestionFormat, AnswerFormat, Format
from ._typing import Questions, Answers


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
    __, input_answer_format = data_format(triplets, answers)
    if answers is None:
        triplets = check_array(triplets, dtype=np.uint32)
    else:
        triplets, answers = check_X_y(triplets, answers, dtype=np.uint32)

    if triplets.shape[1] != 3:
        raise ValueError(f"Expects triplet array with three columns, got shape {triplets.shape}.")

    # repeat questions with answers >1/<-1
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


def check_triplet_questions(triplets: Questions, result_format: Union[str, QuestionFormat, None] = None,
                            n_objects: Optional[int] = None) -> Questions:
    """ Input validation for triplet formats.

    Checks questions and answers for shape and datatype.
    Converts between array (T-STE style) and sparse matrix format for questions.
    For array format, also converts between different answer formats.

    Args:
        triplets: Either array_like with index-questions or sparse matrix.
        answers: Optional answers per index-triplet.
        result_format: One of 'list', or 'tensor'. If none, format is not changed.
        sort_others: If true, then assures that for every triplet (i, j, k): j < k
                     This is ignored for answer_format='order'.
        n_objects: The number of individual objects in questions, optional.
                   If not provided with format='sparse', value is inferred
                   by the cube-root the shape product.

    Returns:
        If answer_format='tensor', a three-dimensional sparse.COO matrix is returned.
        The three dimensions all have size 'n_objects'.
        The entry questions[i, j, k] indicates the answer on ij <= jk.
        It is -1 if wrong, 0 if undecidable, and 1 if correct.

        If answer_format='list',
        a numpy array of shape (n_samples, 3) is returned.
        Each row (i, j, k) indicates, ij <= ik.

    Raises:
        ValueError: If the array_like input has the wrong shape, or answer types cannot be converted.
                    This happens e.g. if undecided (0) answers, should be converted to ordered or boolean answers.
    """
    input_format, __ = data_format(triplets)
    output_format = QuestionFormat(result_format or input_format)
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


def check_triplet_answers(triplet_answers: Union[Questions, Answers], answers: Optional[np.ndarray] = None,
                          result_format: Optional[Format] = None, sort_others: bool = True, n_objects: Optional[int] = None
                          ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """ Input validation for triplet formats.

    Checks questions and answers for shape and datatype.
    Converts between list (T-STE style) and tensor (sparse matrix) format for questions.
    For array format, also converts between different answer formats.

    Args:
        questions: Either array_like with index-questions or sparse matrix.
        answers: Optional answers per index-triplet.
        result_format: Format of result
        sort_others: If true, then assures that for every triplet (i, j, k): j < k
                     This is ignored for format='list-order'.
        n_objects: The number of individual objects in questions, optional.
                   If not provided with format='tensor-count', value is inferred
                   by the cube-root the shape product.

    Returns:
        If format='tensor-count', a three-dimensional sparse.COO matrix is returned.
        The three dimensions all have size 'n_objects'.
        The entry questions[i, j, k] indicates the answer on ij <= jk.
        It is -1 if wrong, 0 if undecidable, and 1 if correct.

        If format='list-order',
        a numpy array of shape (n_samples, 3) is returned.
        Each row (i, j, k) indicates, ij <= ik.

        If format='list-count', two numpy arrays are of shape (n_samples, 3)
        and n_samples are returned.
        The first array contains index-questions (i, j, k).
        The second array elements represent the answer as described above for format='tensor-count'.

        If answer_format='list-boolean', same as for answer_format='list-count'.
        The answers are True/False instead of 1/-1.

    Raises:
        ValueError: If the array_like input has the wrong shape, or answer types cannot be converted.
                    This happens e.g. if undecided (0) answers, should be converted to ordered or boolean answers.
    """
    if isinstance(triplet_answers, tuple) and answers is None:
        triplets, answers = triplet_answers
    else:
        triplets = triplet_answers
    input_question_format, input_answer_format = data_format(triplets, answers)
    output_question_format, output_answer_format = check_format(result_format, triplets, answers)

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
