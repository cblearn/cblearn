import numpy as np

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


def test_triplet_error():
    pass


def test_triplet_scorer():
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    triplets, answers = datasets.triplet_response(triplets, embedding, result_format='list-count')

    estimator = DummyOrdinalEmbedding(embedding)
    assert metrics.QueryScorer(estimator, triplets, answers) == 1
    assert metrics.QueryScorer(estimator, triplets, ~answers) == 0
