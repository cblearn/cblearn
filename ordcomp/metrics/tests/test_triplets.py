import numpy as np

from ordcomp import metrics
from ordcomp import datasets


class DummyOrdinalEmbedding():
    def __init__(self, embedding):
        self.embedding = embedding

    def transform(self, *args, **kwargs):
        return self.embedding

    def predict(self, triplets):
        return datasets._triplet_generation.judge_triplets(triplets, self.embedding, return_responses=True, noise=None)


def test_triplet_error():
    triplets = datasets.make_triplets(10)
    embedding = np.random.random((10, 2))
    responses = datasets._triplet_generation.judge_triplets(triplets, embedding, return_responses=True)
    assert metrics.triplet_error(triplets, embedding, responses=responses) == 0

    triplets = triplets[:, [0, 2, 1]]
    assert metrics.triplet_error(triplets, embedding, responses=responses) == 1


def test_triplet_scorer():
    triplets = datasets.make_triplets(10)
    embedding = np.random.random((10, 2))
    responses = datasets._triplet_generation.judge_triplets(triplets, embedding, return_responses=True)

    estimator = DummyOrdinalEmbedding(embedding)
    assert metrics.TripletScorer(estimator, triplets, responses) == 1
    assert metrics.TripletScorer(estimator, triplets[:, [0, 2, 1]], responses) == 0
