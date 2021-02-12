import numpy as np
import pytest

from cblearn import datasets


def test_make_triplets_raises():
    with pytest.raises(ValueError):
        datasets.make_random_triplet_indices(n_objects=12, size=100000, repeat=False)


def test_make_random_triplets_all():
    triplets = datasets.make_random_triplet_indices(n_objects=12, size=1., make_all=np.inf)
    np.testing.assert_equal(triplets.shape, (660, 3))
    np.testing.assert_array_less(len(np.unique(triplets, axis=0)), 660)

    triplets = datasets.make_random_triplet_indices(n_objects=12, size=1., monotonic=True, make_all=np.inf)
    np.testing.assert_equal(triplets.shape, (220, 3))

    triplets = datasets.make_random_triplet_indices(n_objects=12, size=1., repeat=False, make_all=np.inf)
    np.testing.assert_equal(np.unique(triplets, axis=1).shape, (660, 3))


def test_make_random_triplets_subsample():
    n_triplets = 100

    # we use a large object size here, which would not allow possible all triplets in memory (>> 1TB RAM)
    triplets = datasets.make_random_triplet_indices(n_objects=12000, size=n_triplets, make_all=0)
    np.testing.assert_equal(triplets.shape, (n_triplets, 3))
    triplets = datasets.make_random_triplet_indices(n_objects=12, size=1., make_all=0)
    np.testing.assert_equal(triplets.shape, (660, 3))

    triplets = datasets.make_random_triplet_indices(n_objects=12, size=1., monotonic=True, make_all=0)
    np.testing.assert_equal(triplets.shape, (220, 3))

    # we use a smaller object size here, to make duplicates more likely
    triplets = datasets.make_random_triplet_indices(n_objects=15, size=n_triplets, repeat=False, make_all=0)
    np.testing.assert_equal(np.unique(triplets, axis=1).shape, (n_triplets, 3))
