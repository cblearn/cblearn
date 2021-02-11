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


def check_format(format: Optional[Format], default_questions: Union[Questions, Answers], default_answers: Optional[np.ndarray]
                 ) -> Tuple[QuestionFormat, AnswerFormat]:
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
    if format:
        if isinstance(format, str):
            format_parts = format.split('-')
            return QuestionFormat(format_parts[0]), AnswerFormat(format_parts[1])
        elif isinstance(format, tuple):
            return QuestionFormat(format[0]), AnswerFormat(format[1])
    else:
        return data_format(default_questions, default_answers)


def data_format(questions: Union[Questions, Answers], answers: Optional[np.ndarray] = None
                ) -> Tuple[QuestionFormat, AnswerFormat]:
    """ Extract format of comparison data.

    Args:
        questions: Comparison questions to extract format.
        answers: Comparison answers to extract format.
    Returns:
        Tuple of format identifiers (question, answer)
    Raises:
        TypeError: Invalid type of data.
    """
    if isinstance(questions, tuple) and answers is None:
        questions, answers = questions

    if answers is None:
        if isinstance(questions, (scipy.sparse.spmatrix, sparse.SparseArray)):
            return QuestionFormat.TENSOR, AnswerFormat.COUNT
        elif isinstance(questions, (Sequence, np.ndarray)):
            return QuestionFormat.LIST, AnswerFormat.ORDER
    elif isinstance(answers, (Sequence, np.ndarray)):
        answer_type = np.asarray(answers).dtype
        if answer_type == np.bool:
            return QuestionFormat.LIST, AnswerFormat.BOOLEAN
        elif np.issubdtype(answer_type, np.number):
            return QuestionFormat.LIST, AnswerFormat.COUNT
        else:
            raise TypeError(f"Unknown question/answer format for answer.dtype=={answer_type}.")
    raise TypeError(f"Unknown question/answer format for {type(questions)}/{type(answers)}.")
