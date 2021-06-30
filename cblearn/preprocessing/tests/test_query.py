import numpy as np

from cblearn.preprocessing import triplets_from_multiselect
from cblearn.preprocessing import triplets_from_oddoneout
from cblearn.preprocessing import triplets_from_mostcentral


def test_triplets_from_multiselect():
    multi_data = np.asarray([[0, 1, 2, 3, 4, 5, 6],
                             [10, 11, 12, 13, 14, 15, 16]])
    n_select = 2
    selected = np.asarray([[2, 1], [1, 2]])

    # Use column data to mark selection
    triplets = triplets_from_multiselect(multi_data, n_select, is_ranked=False)
    assert triplets.shape == (len(multi_data) * (4 + 4), 3)
    assert np.isin(triplets[:, 1], [1, 2, 11, 12]).all()
    assert np.isin(triplets[:, 2], multi_data[:, 3:]).all()

    triplets = triplets_from_multiselect(multi_data, n_select, is_ranked=True)
    assert triplets.shape == (len(multi_data) * (5 + 4), 3)
    assert np.isin(triplets[:, 1], [1, 2, 11, 12]).all()
    assert np.isin(triplets[:, 2], multi_data[:, 2:]).all()

    # Use multiple labels to mark selection
    triplets = triplets_from_multiselect(multi_data, selected, is_ranked=False)
    assert triplets.shape == (len(multi_data) * (4 + 4), 3)
    assert np.isin(triplets[:, 1], [1, 2, 11, 12]).all()
    assert np.isin(triplets[:, 2], multi_data[:, 3:]).all()

    triplets = triplets_from_multiselect(multi_data, selected, is_ranked=True)
    print(triplets)
    assert triplets.shape == (len(multi_data) * (5 + 4), 3)
    assert np.isin(triplets[:, 1], [1, 2, 11, 12]).all()
    assert np.isin(triplets[:, 2], [1, 3, 4, 5, 6, 12, 13, 14, 15, 16]).all()


def test_triplets_from_oddoneout():
    oddone_data = np.asarray([[0, 1, 2, 3],
                              [5, 6, 7, 8]])
    selected = np.asarray([0, 2])
    n_triplets = len(oddone_data) * (4 - 1) * (4 - 2)

    triplets = triplets_from_oddoneout(oddone_data)  # select first column
    assert triplets.shape == (n_triplets, 3)
    np.testing.assert_equal(np.unique(triplets[:, 0]), [1, 2, 3, 6, 7, 8])
    np.testing.assert_equal(np.unique(triplets[:, 1]), [1, 2, 3, 6, 7, 8])
    np.testing.assert_equal(np.unique(triplets[:, 2]), [0, 5])

    triplets = triplets_from_oddoneout(oddone_data, selected)
    assert triplets.shape == (n_triplets, 3)
    np.testing.assert_equal(np.unique(triplets[:, 0]), [1, 2, 3, 5, 6, 8])
    np.testing.assert_equal(np.unique(triplets[:, 1]), [1, 2, 3, 5, 6, 8])
    np.testing.assert_equal(np.unique(triplets[:, 2]), [0, 7])


def test_triplets_from_mostcentral():
    mostcentral_data = np.asarray([[0, 1, 2, 3],
                                   [5, 6, 7, 8]])
    selected = np.asarray([0, 2])
    n_triplets = len(mostcentral_data) * (4 - 1) * (4 - 2)

    triplets = triplets_from_mostcentral(mostcentral_data)  # select first column
    assert triplets.shape == (n_triplets, 3)
    np.testing.assert_equal(np.unique(triplets[:, 0]), [1, 2, 3, 6, 7, 8])
    np.testing.assert_equal(np.unique(triplets[:, 1]), [0, 5])
    np.testing.assert_equal(np.unique(triplets[:, 2]), [1, 2, 3, 6, 7, 8])

    triplets = triplets_from_mostcentral(mostcentral_data, selected)
    assert triplets.shape == (n_triplets, 3)
    np.testing.assert_equal(np.unique(triplets[:, 0]), [1, 2, 3, 5, 6, 8])
    np.testing.assert_equal(np.unique(triplets[:, 1]), [0, 7])
    np.testing.assert_equal(np.unique(triplets[:, 2]), [1, 2, 3, 5, 6, 8])
