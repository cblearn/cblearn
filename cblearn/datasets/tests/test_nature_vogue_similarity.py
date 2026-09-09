import numpy as np
import pytest

from cblearn.datasets import fetch_nature_scene_similarity
from cblearn.datasets import fetch_vogue_cover_similarity


@pytest.mark.remote_data
def test_fetch_nature_scene(tmp_path):
    data_home = tmp_path / 'cblearn_datasets'
    bunch = fetch_nature_scene_similarity(data_home=data_home, shuffle=False)

    assert bunch.triplet.shape == (3355, 3)
    assert bunch.image_label.shape == (120, )
    assert bunch.image_label[[0, -1]].tolist() == ['art114.jpg', 'n344019.jpg']

    triplets = fetch_nature_scene_similarity(data_home=data_home, shuffle=False, return_triplets=True)
    np.testing.assert_equal(bunch.triplet, triplets)


@pytest.mark.remote_data
def test_fetch_vogue_cover(tmp_path):
    data_home = tmp_path / 'cblearn_datasets'
    bunch = fetch_vogue_cover_similarity(data_home=data_home, shuffle=False)

    assert bunch.triplet.shape == (1107, 3)
    assert bunch.image_label.shape == (60, )
    assert bunch.image_label[[0, -1]].tolist() == ['Cover_uk_VOgue_MAY10_V_29mar10_bt_268x353.jpg',
                                                   'voguecoverapr11_bt_268x353.jpg']

    triplets = fetch_vogue_cover_similarity(data_home=data_home, shuffle=False, return_triplets=True)
    np.testing.assert_equal(bunch.triplet, triplets)


@pytest.mark.remote_data
def test_both_datasets_share_one_download(tmp_path):
    """ Both fetchers read the same archive, so the second must not re-download. """
    data_home = tmp_path / 'cblearn_datasets'
    fetch_nature_scene_similarity(data_home=data_home, shuffle=False)
    bunch = fetch_vogue_cover_similarity(data_home=data_home, shuffle=False, download_if_missing=False)

    assert bunch.triplet.shape == (1107, 3)


@pytest.mark.remote_data
def test_triplet_indices_are_valid(tmp_path):
    """ Every index must address one of the labelled images. """
    data_home = tmp_path / 'cblearn_datasets'
    for fetch, n_objects in [(fetch_nature_scene_similarity, 120), (fetch_vogue_cover_similarity, 60)]:
        bunch = fetch(data_home=data_home, shuffle=False)
        assert bunch.triplet.min() >= 0
        assert bunch.triplet.max() == n_objects - 1
        assert (bunch.triplet[:, 0] != bunch.triplet[:, 1]).all()
        assert (bunch.triplet[:, 0] != bunch.triplet[:, 2]).all()
        assert (bunch.triplet[:, 1] != bunch.triplet[:, 2]).all()
