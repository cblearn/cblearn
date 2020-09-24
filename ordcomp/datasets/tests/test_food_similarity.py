import numpy as np

from ordcomp.datasets import fetch_food_similarity


def test_fetch_food():
    bunch = fetch_food_similarity(shuffle=False)

    assert bunch.data.shape == (190376, 3)
    assert bunch.image_names.shape == (100, )
    assert bunch.image_names[bunch.data[0, 0]] == 'images/214649bfd7ea489b8daf588e6fed45aa.jpg'

    triplets = fetch_food_similarity(shuffle=False, return_triplets=True)
    np.testing.assert_equal(bunch.data, triplets)
