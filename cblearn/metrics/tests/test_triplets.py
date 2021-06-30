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
        result = datasets.triplet_response(triplets, self.embedding)
        if isinstance(result, tuple):
            return result[1]
        else:
            return result


@pytest.mark.parametrize("format", ['list-order', 'list-count', 'list-boolean', 'tensor-count'])
def test_triplet_error(format):
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    triplets, bool_answers = datasets.triplet_response(triplets, embedding, result_format='list-boolean')
    order_triplets = datasets.triplet_response(triplets, embedding, result_format='list-order')

    test_answers = datasets.triplet_response(triplets, embedding, result_format=format)

    if isinstance(test_answers, tuple):
        test_triplets, test_answers = test_answers
        assert metrics.query_error(test_answers, bool_answers) == 0
        assert metrics.query_error(test_answers, ~bool_answers) == 1
        assert metrics.query_error(test_answers, np.random.permutation(bool_answers)) > 0
        assert metrics.query_error(np.random.permutation(test_answers), bool_answers) > 0

        with pytest.raises(ValueError):
            metrics.query_error((test_triplets, test_answers), (test_triplets + 1, test_answers))
        with pytest.raises(ValueError):
            metrics.query_error((test_triplets, test_answers), bool_answers)
    else:
        assert metrics.query_error(test_answers, test_answers) == 0
        assert metrics.query_error(test_answers, order_triplets) == 0
        assert metrics.query_error(test_answers, embedding) == 0

        with pytest.raises(ValueError):
            metrics.query_error(test_answers, bool_answers) == 1


def test_triplet_scorer():
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    triplets, answers = datasets.triplet_response(triplets, embedding, result_format='list-boolean')

    estimator = DummyOrdinalEmbedding(embedding)
    assert metrics.QueryScorer(estimator, triplets, answers) == 1
    assert metrics.QueryScorer(estimator, triplets, ~answers) == 0
