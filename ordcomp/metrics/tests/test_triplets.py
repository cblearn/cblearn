import numpy as np

from ordcomp import metrics
from ordcomp import datasets


class DummyOrdinalEmbedding():
    def __init__(self, embedding):
        self.embedding = embedding

    def transform(self, *args, **kwargs):
        return self.embedding

    def predict(self, triplets):
        return datasets.triplet_answers(triplets, self.embedding)


def test_triplet_error():
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    triplets, answers = datasets.triplet_answers(triplets, embedding, question_format='list', answer_format='boolean')
    assert metrics.triplet_error((triplets, answers), embedding) == 0
    assert metrics.triplet_error((triplets, ~answers), embedding) == 1
    assert metrics.triplet_error((triplets, answers), (triplets, answers)) == 0
    assert metrics.triplet_error((triplets, answers), (triplets, answers)) == 0


def test_triplet_scorer():
    triplets = datasets.make_random_triplet_indices(10)
    embedding = np.random.random((10, 2))
    triplets, answers = datasets.triplet_answers(triplets, embedding, question_format='list', answer_format='boolean')

    estimator = DummyOrdinalEmbedding(embedding)
    assert metrics.TripletScorer(estimator, triplets, (triplets, answers)) == 1
    assert metrics.TripletScorer(estimator, triplets, (triplets, ~answers)) == 0
