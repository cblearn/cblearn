import numpy as np
import pytest

from cblearn.datasets import fetch_things_similarity


@pytest.mark.remote_data
def test_fetch_things_word_id_order():
    data = fetch_things_similarity()

    uid = data.uid
    for f, t in (('_', ' '), ('1', ''), ('2', ''), ('3', ''), ('4', ''), ('flip flop', 'flip-flop')):
        # manually convert ids to words by simple replacements
        uid = np.array([w.replace(f, t) for w in uid])
    mask = uid != data.word
    np.testing.assert_equal(uid, data.word,
                            f"Expects same order of word uid and image uid. {list(zip(uid[mask], data.word[mask]))}")
