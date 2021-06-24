import numpy as np
import pytest

from cblearn import metrics
from cblearn import datasets


class DummyOrdinalEmbedding():
    def __init__(self, embedding):
        self.embedding = embedding

    def transform(self, *args, **kwargs):
        return self.embedding

    def predict(self, triplets):
        return datasets.triplet_answers(triplets, self.embedding)


@pytest.mark.parametrize("format", ['list-order', 'list-count', 'list-boolean', 'tensor-count'])
def test_triplet_error(format):
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    triplets, bool_answers = datasets.triplet_answers(triplets, embedding, result_format='list-boolean')

    test_answers = datasets.triplet_answers(triplets, embedding, result_format=format)
    assert metrics.triplet_error(test_answers, bool_answers) == 0
    assert metrics.triplet_error(test_answers, ~bool_answers) == 1
    assert metrics.triplet_error(test_answers, embedding) == 0
    assert metrics.triplet_error(test_answers, test_answers) == 0
    assert metrics.triplet_error(test_answers, np.random.permutation(bool_answers)) > 0

    if isinstance(test_answers, tuple):
        test_triplets, test_answers = test_answers
        assert metrics.triplet_error(test_answers, bool_answers) == 0
        assert metrics.triplet_error((test_triplets, np.random.permutation(test_answers)), embedding) > 0

        with pytest.raises(ValueError):
            metrics.triplet_error((test_triplets, test_answers), (test_triplets + 1, test_answers))



def test_triplet_scorer():
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    triplets, answers = datasets.triplet_answers(triplets, embedding, result_format='list-boolean')

    estimator = DummyOrdinalEmbedding(embedding)
    assert metrics.TripletScorer(estimator, triplets, (triplets, answers)) == 1
    assert metrics.TripletScorer(estimator, triplets, (triplets, ~answers)) == 0
