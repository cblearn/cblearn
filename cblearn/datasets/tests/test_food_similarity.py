import numpy as np
import pytest

from cblearn.datasets import fetch_food_similarity


@pytest.mark.remote_data
def test_fetch_food(tmp_path):
    data_home = tmp_path / 'cblearn_datasets'
    bunch = fetch_food_similarity(data_home=data_home, shuffle=False)

    assert bunch.data.shape == (190376, 3)
    assert bunch.image_names.shape == (100, )
    assert bunch.image_names[bunch.data[0, 0]] == 'images/214649bfd7ea489b8daf588e6fed45aa.jpg'

    triplets = fetch_food_similarity(data_home=data_home, shuffle=False, return_triplets=True)
    np.testing.assert_equal(bunch.data, triplets)
