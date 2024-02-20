import pytest
import numpy as np

from cblearn.datasets import triplet_response


def test_triplet_response_validates_input():
    n = 5  # n objects
    t = 10  # n triplets
    d = 2  # n dimensions
    valid_queries = [
        np.random.choice(n, size=3, replace=False)
        for _ in range(t)
    ]
    invalid_queries_1 = [
        np.random.choice(n, size=5, replace=False)
        for _ in range(t)
    ]
    invalid_queries_2 = [
        np.random.choice(n + 1, size=3, replace=False)
        for _ in range(t)
    ]
    invalid_queries_3 = np.random.uniform(low=-1, high=1, size=(t, 3))
    embedding = np.random.normal(size=(n, d))

    responses = triplet_response(valid_queries, embedding)
    assert responses.shape == (t, 3)
    with pytest.raises(ValueError):
        triplet_response(invalid_queries_1, embedding)
    with pytest.raises(ValueError):
        triplet_response(invalid_queries_2, embedding)
    with pytest.raises(ValueError):
        triplet_response(invalid_queries_3, embedding)