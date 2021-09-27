import numpy as np
import pytest

from cblearn.datasets import fetch_musician_similarity


@pytest.mark.remote_data
def test_fetch_musician_similarity(tmp_path):
    data_home = tmp_path / 'cblearn_datasets'
    bunch = fetch_musician_similarity(data_home=data_home, shuffle=False)

    assert bunch.data.shape == (131_970, 3)
    assert bunch.judgement_id.shape == (131_970, )
    assert bunch.user.shape == (131_970, )
    assert bunch.survey_or_game.shape == (131_970, )
    assert bunch.artist_name.shape == (448, )
    assert bunch.artist_id.shape == (448, )
    assert bunch.artist_name[bunch.data][0, 0] == 'queen'
    assert tuple(bunch.artist_id[bunch.data][0]) == (4325, 1735, 3295)
    assert tuple(bunch.artist_id[bunch.data][-1]) == (3603, 4913, 4948)

    triplets = fetch_musician_similarity(data_home=data_home, shuffle=False, return_triplets=True)
    np.testing.assert_equal(bunch.data, triplets)
    np.testing.assert_equal(bunch.artist_name[triplets], bunch.artist_name[bunch.data])

    shuffle_bunch = fetch_musician_similarity(data_home=data_home, random_state=42)
    assert not np.all(shuffle_bunch.data == bunch.data)
    assert not np.all(shuffle_bunch.user == bunch.user)
    np.testing.assert_equal(shuffle_bunch.user.sort(), bunch.user.sort())