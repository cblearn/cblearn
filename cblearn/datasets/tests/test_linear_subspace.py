import numpy as np
import pytest
from cblearn.datasets import LinearSubspace
from functools import partial
from scipy.spatial.distance import pdist, squareform

# Subspace Dimension and Space Dimension tests


def test_subspace_dimension_1():
    manifold = LinearSubspace(subspace_dimension=1, space_dimension=3)
    assert manifold.subspace_dimension == 1
    points, _ = manifold.sample_points(num_points=10)
    assert points.shape == (10, 3)
    assert manifold.basis.shape == (2, 3)


def subspace_dimension_greater_than_space_dimension(subspace_dimension, space_dimension, random_state):
    _ = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=random_state)
    # Add your actual test logic here


@pytest.mark.parametrize("subspace_dimension, space_dimension, random_state", [
    (1, 1, 1),
    (1, 0, 1),
    (2, 1, 1),
    (4, 3, 1),
])
def test_subspace_dimension_greater_than_space_dimension(subspace_dimension, space_dimension, random_state):
    with pytest.raises(ValueError):
        subspace_dimension_greater_than_space_dimension(subspace_dimension, space_dimension, random_state)


def invalid_subspace_dimension(subspace_dimension):
    _ = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=3)


@pytest.mark.parametrize("subspace_dimension", [-1, 0, "2", 1.5]
                         )
def test_invalid_subspace_dimension(subspace_dimension):
    with pytest.raises(ValueError):
        invalid_subspace_dimension(subspace_dimension)


def invalid_space_dimension(space_dimension):
    _ = LinearSubspace(subspace_dimension=1, space_dimension=space_dimension)


@pytest.mark.parametrize("space_dimension", [-1, 0, "2", 1.5]
                         )
def test_invalid_space_dimension(space_dimension):
    with pytest.raises(ValueError):
        invalid_space_dimension(space_dimension)

# # Noise and Sampling Function Tests


@pytest.mark.parametrize("subspace_dimension, space_dimension", [
    (1, 2),
    (2, 2),
    (4, 4),
    (1, 3),
    (3, 10)
])
def test_no_random_state(subspace_dimension, space_dimension):
    manifold = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=None)
    points, _ = manifold.sample_points(num_points=10)
    assert points.shape == (10, space_dimension)


@pytest.mark.parametrize("subspace_dimension, space_dimension, random_state", [
    (1, 2, 1),
    (2, 2, 2),
    (4, 4, 10),
    (1, 3, 1),
    (3, 10, 1)
])
def test_no_noise(subspace_dimension, space_dimension, random_state):
    manifold1 = LinearSubspace(
        subspace_dimension=subspace_dimension,
        space_dimension=space_dimension,
        random_state=random_state)
    manifold1._create_manifold()
    manifold2 = LinearSubspace(
        subspace_dimension=subspace_dimension,
        space_dimension=space_dimension,
        random_state=random_state)
    manifold2._create_manifold()
    # Check that basis is the same
    np.testing.assert_array_equal(manifold1.basis, manifold2.basis)


@pytest.mark.parametrize("subspace_dimension, space_dimension, noise", [
    (1, 2, 'normal'),
    (2, 2, 'laplace'),
    (4, 4, 'normal'),
    (1, 3, 'gumbel'),
    (3, 10, 'normal')
])
def test_noise_string(subspace_dimension, space_dimension, noise):
    manifold = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=1)
    points = manifold.sample_points(num_points=100, noise=noise)
    assert points[0].shape == (100, space_dimension)


def randint_wrapper(low, high, size):
    return np.random.randint(low=low, high=high, size=size)


@pytest.mark.parametrize("subspace_dimension, space_dimension, noise", [
    (1, 2, np.random.normal),
    (2, 2, partial(randint_wrapper, low=1, high=10)),
    (4, 4, np.random.normal),
    (1, 3, partial(randint_wrapper, low=1, high=10)),
    (3, 10, np.random.gumbel)
])
def test_noise_callable(subspace_dimension, space_dimension, noise):
    manifold = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=1)
    points = manifold.sample_points(num_points=100, noise=noise)
    assert points[0].shape == (100, space_dimension)


@pytest.mark.parametrize("subspace_dimension, space_dimension, sampling_function, random_state", [
    (1, 2, 'normal', 1),
    (2, 2, 'normal', 1),
    (4, 4, 'normal', 1),
    (1, 3, 'normal', 1),
    (3, 10, 'normal', 1)
])
def test_sampling_function(subspace_dimension, space_dimension, sampling_function, random_state):
    manifold1 = LinearSubspace(
        subspace_dimension=subspace_dimension,
        space_dimension=space_dimension,
        random_state=random_state)
    points1 = manifold1.sample_points(num_points=100, sampling_function=sampling_function, random_state=random_state)
    assert points1[0].shape == (100, space_dimension)
    manifold2 = LinearSubspace(
        subspace_dimension=subspace_dimension,
        space_dimension=space_dimension,
        random_state=random_state)
    points2 = manifold2.sample_points(num_points=100, sampling_function=sampling_function, random_state=random_state)
    np.testing.assert_array_equal(points1[0], points2[0])

# # Test distances or not


@pytest.mark.parametrize("subspace_dimension, space_dimension, return_distances", [
    (1, 2, False),
    (2, 2, False),
    (4, 4, False),
    (1, 3, False),
    (3, 10, False)
])
def test_sample_points_return_no_distances(subspace_dimension, space_dimension, return_distances):
    manifold = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=1)
    points = manifold.sample_points(num_points=100, return_distances=return_distances)
    assert points.shape == (100, space_dimension)


@pytest.mark.parametrize("subspace_dimension, space_dimension, return_distances", [
    (1, 2, True),
    (2, 2, True),
    (4, 4, True),
    (1, 3, True),
    (3, 10, True)
])
def test_sample_points_return_distances(subspace_dimension, space_dimension, return_distances):
    manifold = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=1)
    points, distances = manifold.sample_points(num_points=100, return_distances=return_distances)
    assert points.shape == (100, space_dimension)
    assert distances.shape == (100, 100)
    assert np.allclose(distances, squareform(pdist(points)))


@pytest.mark.parametrize("subspace_dimension, space_dimension, return_distances, noise", [
    (1, 2, True, 'normal'),
    (2, 2, True, 'normal'),
    (4, 4, True, 'normal'),
    (1, 3, True, 'normal'),
    (3, 10, True, 'normal')
])
def test_canonical_distance_matrix(subspace_dimension, space_dimension, return_distances, noise):
    manifold = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=1)
    points, distances = manifold.sample_points(num_points=100, return_distances=return_distances, noise=noise)
    assert np.allclose(distances, manifold.get_canonical_distance_matrix(points))


# # Test saving, loading and cloning

@pytest.mark.parametrize("subspace_dimension, space_dimension, random_state", [
    (1, 2, 1),
    (2, 2, 2),
    (4, 4, 10),
    (1, 3, 1),
    (3, 10, 1)
])
def test_set_get_params(subspace_dimension, space_dimension, random_state):
    manifold1 = LinearSubspace(
        subspace_dimension=subspace_dimension,
        space_dimension=space_dimension,
        random_state=random_state)
    manifold1._create_manifold()
    manifold2 = LinearSubspace(subspace_dimension=subspace_dimension, space_dimension=space_dimension, random_state=3)
    manifold2._create_manifold()
    params = manifold1.get_params()
    manifold2.set_params(params)
    np.testing.assert_equal(manifold1.basis, manifold2.basis)


@pytest.mark.parametrize("subspace_dimension, space_dimension, random_state", [
    (1, 2, 1),
    (2, 2, 2),
    (4, 4, 10),
    (1, 3, 1),
    (3, 10, 1)
])
def test_clone(subspace_dimension, space_dimension, random_state):
    manifold = LinearSubspace(
        subspace_dimension=subspace_dimension,
        space_dimension=space_dimension,
        random_state=random_state)
    manifold._create_manifold()
    manifold_clone = manifold.clone()
    np.testing.assert_equal(manifold.basis, manifold_clone.basis)
