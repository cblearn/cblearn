import numpy as np

from ordcomp import metrics
from ordcomp import datasets


class DummyOrdinalEmbedding():
    def __init__(self, embedding):
        self.embedding = embedding

    def transform(self, *args, **kwargs):
        return self.embedding

    def predict(self, triplets):
        return datasets.triplet_responses(triplets, embedding=self.embedding)


def test_triplet_error():
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    responses = datasets.triplet_responses(triplets, embedding=embedding)
    assert metrics.triplet_error(triplets, embedding, responses=responses) == 0

    triplets = triplets[:, [0, 2, 1]]
    assert metrics.triplet_error(triplets, embedding, responses=responses) == 1


def test_triplet_scorer():
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    responses = datasets.triplet_responses(triplets, embedding=embedding)

    estimator = DummyOrdinalEmbedding(embedding)
    assert metrics.TripletScorer(estimator, triplets, responses) == 1
    assert metrics.TripletScorer(estimator, triplets[:, [0, 2, 1]], responses) == 0
