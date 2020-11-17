import pytest
import numpy as np
import sparse

from ordcomp import utils


triplets_numeric_undecided = [[0, 1, 2],
                              [0, 1, 2],
                              [0, 1, 2],
                              [3, 0, 2],
                              [1, 2, 2]]
answers_numeric_undecided = [1, 1, 1, -1, 0]

triplets_explicit = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [3, 0, 2]]
answers_numeric = [1, 1, 1, -1]
answers_binary = [True, True, True, False]
triplets_ordered = [[0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [3, 2, 0]]

triplets_spmatrix = sparse.COO(np.transpose(triplets_explicit), answers_numeric, shape=(4, 4, 4))


def test_check_triplet_questions():
    """ Test the conversation between array and matrix format for questions. """
    triplets = utils.check_triplet_questions(triplets_ordered)
    np.testing.assert_equal(triplets, triplets_ordered)

    triplets = utils.check_triplet_questions(triplets_ordered, format='tensor')
    np.testing.assert_equal(triplets, triplets_spmatrix)
    np.testing.assert_equal(np.triu(triplets), triplets)

    triplets = utils.check_triplet_questions(triplets_spmatrix, format='list')
    np.testing.assert_equal(triplets, triplets_ordered)


def test_check_triplet_answers():
    """ Test the conversation between array and matrix format for question+answers. """
    triplets, answers = utils.check_triplet_answers(triplets_ordered, answer_format='count')
    np.testing.assert_equal(triplets, triplets_explicit)
    np.testing.assert_equal(answers, answers_numeric)

    with pytest.raises(ValueError):
        utils.check_triplet_answers(np.asarray(triplets_ordered)[:, :2], answer_format='count')

    triplets = utils.check_triplet_answers(triplets_ordered, question_format='tensor', answer_format='count')
    np.testing.assert_equal(triplets, triplets_spmatrix)

    # spmatrix contains duplicates, which have to be unrolled for the array format
    triplets = utils.check_triplet_answers(triplets_spmatrix, question_format='list', answer_format='order')
    np.testing.assert_equal(triplets, triplets_ordered)

    # conversation works, if duplicates are dropped.
    triplets = utils.check_triplet_answers(triplets_spmatrix.clip(-1, 1), question_format='list', answer_format='order')
    np.testing.assert_equal(triplets, np.unique(triplets_ordered, axis=0))

    triplets = utils.check_triplet_answers(triplets_spmatrix.reshape((4, 16)).tocsr())
    np.testing.assert_equal(triplets, triplets_spmatrix)


def test_check_triplet_answers_sort_others():
    triplets, answers = utils.check_triplet_answers(triplets_ordered, sort_others=False, answer_format='boolean')
    assert np.all(answers)
    assert not np.all(triplets[:, 1] <= triplets[:, 2])

    triplets, answers = utils.check_triplet_answers((triplets, answers), sort_others=True, answer_format='boolean')
    assert not np.all(answers)
    assert np.all(triplets[:, 1] <= triplets[:, 2])

    triplets = utils.check_triplet_answers(triplets_ordered, sort_others=False,
                                           question_format='tensor', answer_format='count')
    assert np.all(triplets.data > 0)
    assert not np.all(triplets.coords[1, :] <= triplets.coords[2, :])

    triplets = utils.check_triplet_answers(triplets_ordered, sort_others=True,
                                           question_format='tensor', answer_format='count')
    assert not np.all(triplets.data > 0)
    assert np.all(triplets.coords[1, :] <= triplets.coords[2, :])


@pytest.mark.parametrize("input",
                         [triplets_ordered,
                          (triplets_explicit, answers_binary),
                          (triplets_explicit, answers_numeric)])
@pytest.mark.parametrize("answer_format,test_output",
                         [('order', triplets_ordered),
                          ('boolean', (triplets_explicit, answers_binary)),
                          ('count', (triplets_explicit, answers_numeric))])
def test_check_triplet_array_answer_format(input, answer_format, test_output):
    """ Test all possible conversations of answer types. """
    if isinstance(input, tuple):
        triplets, answers = input
    else:
        triplets, answers = input, None
    triplet_answers = utils.transform._check_triplet_array(triplets, answers, sort_others=True,
                                                           answer_format=utils.AnswerFormat(answer_format))
    np.testing.assert_equal(triplet_answers, test_output)


def test_check_triplet_array_answer_undecided():
    with pytest.raises(ValueError):
        utils.transform._check_triplet_array(triplets_numeric_undecided, answers_numeric_undecided,
                                             sort_others=True, answer_format=utils.AnswerFormat.ORDER)
    with pytest.raises(ValueError):
        utils.transform._check_triplet_array(triplets_numeric_undecided, answers_numeric_undecided,
                                             sort_others=True, answer_format=utils.AnswerFormat.BOOLEAN)
    triplets, answers = utils.transform._check_triplet_array(triplets_numeric_undecided, answers_numeric_undecided,
                                                             sort_others=True, answer_format=utils.AnswerFormat.COUNT)
    np.testing.assert_equal(triplets, triplets_numeric_undecided)
    np.testing.assert_equal(answers, answers_numeric_undecided)


def test_check_size():
    assert 6 == utils.check_size(None, 6)
    assert 3 == utils.check_size(3, 6)
    assert 3 == utils.check_size(3., 6)
    assert 3 == utils.check_size(.5, 6)
    assert 15 == utils.check_size(15, 6)

    with pytest.raises(ValueError):
        utils.check_size(-1, 6)
