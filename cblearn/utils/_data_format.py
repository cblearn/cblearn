import enum
from typing import Union, Optional, Tuple, Sequence

import scipy
import sparse
import numpy as np

from ._typing import Questions, Answers


class QuestionFormat(enum.Enum):
    LIST = 'list'
    TENSOR = 'tensor'


class AnswerFormat(enum.Enum):
    ORDER = 'order'
    BOOLEAN = 'boolean'
    COUNT = 'count'


Format = Union[str, Tuple[QuestionFormat, AnswerFormat]]


def check_format(format: Optional[Format], default_questions: Union[Questions, Answers],
                 default_answers: Optional[np.ndarray]) -> Tuple[QuestionFormat, AnswerFormat]:
    """ Validate comparison format description.

    The format of comparison data is specified as a single string separated by '-', or a tuple of format identifiers.
    Valid formats include 'list-order', 'list-boolean', 'list-count', and 'tensor-count'.
    If the format is not explicitly given, the format used by the default_triplets and default_answers is returned.

    Args:
        format: Comparison format string 'question-answer' or tuple of format strings (question, answer).
        default_questions: Comparison questions to extract format from if not passed explicitly.
        default_answers: Comparison answers to extract format from if not passed explicitly.
    Returns:
        Tuple of format identifiers (question, answer)
    Raises:
        ValueError: Format identifier is unknown.
        IndexError: Less than 2 format components.
    """
    if format is None:
        return data_format(default_questions, default_answers)
    else:
        if isinstance(format, str):
            format_parts = format.split('-')
            return QuestionFormat(format_parts[0]), AnswerFormat(format_parts[1])
        elif isinstance(format, tuple):
            return QuestionFormat(format[0]), AnswerFormat(format[1])
        else:
            raise ValueError(f"Expects either format as string, enum-tuple or None; got {format}.")


def data_format(query: Union[Questions], response: Optional[np.ndarray] = None
                ) -> Tuple[QuestionFormat, AnswerFormat]:
    """ Extract format of comparison data.

    Args:
        query: Comparison questions to extract format.
        response: Comparison answers to extract format.
    Returns:
        Tuple of format identifiers (question, answer)
    Raises:
        TypeError: Invalid type of data.
    """
    if isinstance(query, (scipy.sparse.spmatrix, sparse.SparseArray)):
        query_format = QuestionFormat.TENSOR
    elif isinstance(query, (Sequence, np.ndarray)):
        query_format = QuestionFormat.LIST
    elif query is None:
        query_format = None
    else:
        raise ValueError(f"Expects query as sequence, array, or sparse array; got {query}")

    if response is None:
        if query_format is QuestionFormat.TENSOR:
            response_dtype = query.dtype
        elif query_format is QuestionFormat.LIST:
            return query_format, AnswerFormat.ORDER
        else:
            return query_format, None
    elif isinstance(response, (Sequence, np.ndarray)):
        response_dtype = np.asarray(response).dtype
    else:
        return query_format, None

    if response_dtype == bool:
        return query_format, AnswerFormat.BOOLEAN
    elif np.issubdtype(response_dtype, np.number):
        return query_format, AnswerFormat.COUNT
    else:
        raise ValueError(f"Expects response dtype bool or numeric, got {response_dtype}")