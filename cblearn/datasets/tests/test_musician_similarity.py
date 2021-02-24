import numpy as np
import pytest

from cblearn.datasets import fetch_musician_similarity


@pytest.mark.download
def test_fetch_musician_similarity(tmp_path):
    data_home = tmp_path / 'cblearn_datasets'
    bunch = fetch_musician_similarity(data_home=data_home, shuffle=False)

    assert bunch.data.shape == (213629, 3)
    assert bunch.judgement_id.shape == (213629, )
    assert bunch.user.shape == (213629, )
    assert bunch.survey_or_game.shape == (213629, )
    assert bunch.artists.shape == (413, )
    assert bunch.artists[bunch.data[0, 0]] == 'queen'

    triplets = fetch_musician_similarity(data_home=data_home, shuffle=False, return_triplets=True)
    np.testing.assert_equal(bunch.data, triplets)

    shuffle_bunch = fetch_musician_similarity(data_home=data_home, random_state=42)
    assert not np.all(shuffle_bunch.data == bunch.data)
    assert not np.all(shuffle_bunch.user == bunch.user)
    np.testing.assert_equal(shuffle_bunch.user.sort(), bunch.user.sort())
