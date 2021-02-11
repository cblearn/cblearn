import numpy as np

from cblearn import metrics


def test_procrustes_distance():
    embedding = np.random.random((10, 2))
    shift_embedding = embedding + 3
    np.testing.assert_almost_equal(metrics.procrustes_distance(embedding, shift_embedding), 0)

    theta = np.radians(60)
    rotate_embedding = embedding @ np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    np.testing.assert_almost_equal(metrics.procrustes_distance(embedding, rotate_embedding), 0)

    scale_embedding = embedding * 7.5
    np.testing.assert_almost_equal(metrics.procrustes_distance(embedding, scale_embedding), 0)

    sheared_embeddign = embedding @ np.array(((1, 5), (0, 1)))
    assert 0.1 < metrics.procrustes_distance(embedding, sheared_embeddign)
