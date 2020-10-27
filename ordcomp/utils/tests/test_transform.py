import pytest
import numpy as np
import sparse

from ordcomp import utils


triplets_numeric_undecided = [[0, 1, 2],
                              [0, 1, 2],
                              [0, 1, 2],
                              [3, 0, 2],
                              [1, 2, 2]]
responses_numeric_undecided = [1, 1, 1, -1, 0]

triplets_explicit = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [3, 0, 2]]
responses_numeric = [1, 1, 1, -1]
responses_binary = [True, True, True, False]
triplets_implicit = [[0, 1, 2],
                     [0, 1, 2],
                     [0, 1, 2],
                     [3, 2, 0]]

triplets_spmatrix = sparse.COO(np.transpose(triplets_explicit), responses_numeric, shape=(4, 4, 4))


def test_check_triplets():
    """ Test the conversation between array and matrix format. """
    triplets, responses = utils.check_triplets(triplets_implicit, format='array', response_type='numeric')
    np.testing.assert_equal(triplets, triplets_explicit)
    np.testing.assert_equal(responses, responses_numeric)

    with pytest.raises(ValueError):
        utils.check_triplets(np.asarray(triplets_implicit)[:, :2], format='array', response_type='numeric')

    triplets = utils.check_triplets(triplets_implicit, format='spmatrix')
    np.testing.assert_equal(triplets, triplets_spmatrix)

    # spmatrix contains duplicates, which currently cannot be converted to implicit array
    with pytest.raises(ValueError):
        utils.check_triplets(triplets_spmatrix, format='array', response_type='implicit')
    # conversation works, if duplicates are dropped.
    triplets = utils.check_triplets(triplets_spmatrix.clip(-1, 1), format='array', response_type='implicit')
    np.testing.assert_equal(triplets, np.unique(triplets_implicit, axis=0))

    triplets = utils.check_triplets(triplets_spmatrix.reshape((4, 16)).tocsr(), format='spmatrix')
    np.testing.assert_equal(triplets, triplets_spmatrix)


@pytest.mark.parametrize("input_triplets,input_responses",
                         [(triplets_implicit, None),
                          (triplets_explicit, responses_binary),
                          (triplets_explicit, responses_numeric)])
@pytest.mark.parametrize("response_type,test_triplets,test_responses",
                         [('implicit', triplets_implicit, None),
                          ('boolean', triplets_explicit, responses_binary),
                          ('numeric', triplets_explicit, responses_numeric)])
def test_check_triplet_array_response_type(input_triplets, input_responses, response_type, test_triplets, test_responses):
    """ Test all possible conversations of response types. """
    triplets, responses = utils.check_triplet_array(input_triplets, input_responses, response_type=response_type)
    np.testing.assert_equal(triplets, test_triplets)
    np.testing.assert_equal(responses, test_responses)


def test_check_triplet_array_response_undecided():
    with pytest.raises(ValueError):
        utils.check_triplet_array(triplets_numeric_undecided, responses_numeric_undecided, response_type='implicit')
    with pytest.raises(ValueError):
        utils.check_triplet_array(triplets_numeric_undecided, responses_numeric_undecided, response_type='boolean')
    triplets, responses = utils.check_triplet_array(triplets_numeric_undecided, responses_numeric_undecided,
                                                    response_type='numeric')
    np.testing.assert_equal(triplets, triplets_numeric_undecided)
    np.testing.assert_equal(responses, responses_numeric_undecided)


def test_check_size():
    assert 6 == utils.check_size(None, 6)
    assert 3 == utils.check_size(3, 6)
    assert 3 == utils.check_size(3., 6)
    assert 3 == utils.check_size(.5, 6)
    assert 15 == utils.check_size(15, 6)

    with pytest.raises(ValueError):
        utils.check_size(-1, 6)
