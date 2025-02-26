import numpy as np
import pytest
from cblearn.datasets import Line
from cblearn.datasets import make_random_triplet_indices
from functools import partial
from scipy.spatial.distance import pdist, squareform

# Space Dimension tests


def invalid_space_dimension(space_dimension):
    _ = Line(space_dimension=space_dimension)


@pytest.mark.parametrize("space_dimension", [-1, 0, "2", 1.5])
def test_invalid_space_dimension(space_dimension):
    with pytest.raises(ValueError):
        invalid_space_dimension(space_dimension)


# Test that Points are actually on a line


@pytest.mark.parametrize(
    "space_dimension, random_state, sampling_function",
    [(1, 1, "normal"), (2, 2, "gumbel"), (4, 3, "laplace"), (3, 4, "uniform"), (10, 5, "normal")],
)
def test_points_on_line(space_dimension, random_state, sampling_function):
    manifold = Line(space_dimension=space_dimension, random_state=random_state)
    points = manifold.sample_points(
        num_points=100,
        sampling_function=sampling_function,
        sampling_options={},
        noise=None,
        random_state=random_state,
        return_distances=False,
    )
    # Check that points are on the line
    for point in points:
        vector_to_point = point - manifold.origin
        projection_length = np.dot(vector_to_point, manifold.direction)
        projected_point = manifold.origin + projection_length * manifold.direction
        np.testing.assert_allclose(point, projected_point, atol=1e-8)


# # Noise and Sampling Function Tests
@pytest.mark.parametrize("space_dimension", [2, 2, 4, 3, 10])
def test_no_random_state(space_dimension):
    manifold = Line(space_dimension=space_dimension, random_state=None)
    points, _ = manifold.sample_points(num_points=10)
    assert points.shape == (10, space_dimension)


@pytest.mark.parametrize("space_dimension, random_state", [(2, 1), (2, 2), (4, 10), (3, 1), (10, 1)])
def test_no_noise(space_dimension, random_state):
    manifold1 = Line(space_dimension=space_dimension, random_state=random_state)
    manifold1._create_manifold()
    manifold2 = Line(space_dimension=space_dimension, random_state=random_state)
    manifold2._create_manifold()
    # Check that basis is the same
    np.testing.assert_array_equal(manifold1.origin, manifold2.origin)
    np.testing.assert_array_equal(manifold1.direction, manifold2.direction)


@pytest.mark.parametrize(
    "space_dimension, noise", [(2, "normal"), (2, "laplace"), (4, "normal"), (3, "gumbel"), (10, "normal")]
)
def test_noise_string(space_dimension, noise):
    manifold = Line(space_dimension=space_dimension, random_state=1)
    points = manifold.sample_points(num_points=100, noise=noise)
    assert points[0].shape == (100, space_dimension)


def randint_wrapper(low, high, size):
    return np.random.randint(low=low, high=high, size=size)


@pytest.mark.parametrize(
    "space_dimension, noise",
    [
        (2, np.random.normal),
        (2, partial(randint_wrapper, low=1, high=10)),
        (4, np.random.normal),
        (3, partial(randint_wrapper, low=1, high=10)),
        (10, np.random.gumbel),
    ],
)
def test_noise_callable(space_dimension, noise):
    manifold = Line(space_dimension=space_dimension, random_state=1)
    points = manifold.sample_points(num_points=100, noise=noise)
    assert points[0].shape == (100, space_dimension)


@pytest.mark.parametrize(
    "space_dimension, noise, random_state",
    [(2, "normal", 1), (2, "normal", 1), (4, "normal", 1), (3, "normal", 1), (10, "normal", 1)],
)
def test_sampling_function(space_dimension, noise, random_state):
    manifold1 = Line(space_dimension=space_dimension, random_state=random_state)
    points1 = manifold1.sample_points(num_points=100, noise=noise, random_state=random_state)
    assert points1[0].shape == (100, space_dimension)
    manifold2 = Line(space_dimension=space_dimension, random_state=random_state)
    points2 = manifold2.sample_points(num_points=100, noise=noise, random_state=random_state)
    np.testing.assert_array_equal(points1[0], points2[0])


# # Test distances or not

@pytest.mark.parametrize("space_dimension, return_distances", [(2, False), (2, False), (4, False), (3, False), (10, False)])
def test_sample_points_return_no_distances(space_dimension, return_distances):
    manifold = Line(space_dimension=space_dimension, random_state=1)
    points = manifold.sample_points(num_points=100, return_distances=return_distances)
    assert points.shape == (100, space_dimension)


@pytest.mark.parametrize("space_dimension, return_distances", [(2, True), (2, True), (4, True), (3, True), (10, True)])
def test_sample_points_return_distances(space_dimension, return_distances):
    manifold = Line(space_dimension=space_dimension, random_state=1)
    points, distances = manifold.sample_points(num_points=100, return_distances=return_distances)
    assert points.shape == (100, space_dimension)
    assert distances.shape == (100, 100)
    assert np.allclose(distances, squareform(pdist(points)))


@pytest.mark.parametrize(
    "space_dimension, return_distances, noise",
    [(2, True, "normal"), (2, True, "normal"), (4, True, "normal"), (3, True, "normal"), (10, True, "normal")],
)
def test_canonical_distance_matrix(space_dimension, return_distances, noise):
    manifold = Line(space_dimension=space_dimension, random_state=1)
    points, distances = manifold.sample_points(num_points=100, return_distances=return_distances, noise=noise)
    assert np.allclose(distances, manifold.get_canonical_distance_matrix(points))


# Test triplet sampling directly
@pytest.mark.parametrize(
    "space_dimension, sampling_function, rs",
    [(2, "normal", 1), (2, "gumbel", 2), (4, "poisson", 3), (3, "exponential", 4), (10, "normal", 5)],
)
def test_sample_triplets(space_dimension, sampling_function, rs):
    manifold = Line(space_dimension=space_dimension, random_state=rs)
    points, distance_matrix, triplets = manifold.sample_triplets(
        num_points=20, num_triplets=100, sampling_function=sampling_function, sampling_options={}, random_state=rs
    )
    # Basic checks
    assert points.shape == (20, space_dimension)
    assert triplets.shape == (100, 3)
    assert np.allclose(distance_matrix, manifold.get_canonical_distance_matrix(points))
    # Random State and Equivalency Checks
    manifold2 = Line(space_dimension=space_dimension, random_state=rs)
    points2, distance_matrix2 = manifold2.sample_points(
        num_points=20, sampling_function=sampling_function, sampling_options={}, random_state=rs
    )
    triplets2 = make_random_triplet_indices(n_objects=20, size=100, random_state=rs)
    assert np.allclose(points[0], points2[0])
    assert np.allclose(distance_matrix, distance_matrix2)
    assert np.allclose(triplets, triplets2)


# # Test saving, loading and cloning


@pytest.mark.parametrize("space_dimension, random_state", [(2, 1), (2, 2), (4, 10), (3, 1), (10, 1)])
def test_set_get_params(space_dimension, random_state):
    manifold1 = Line(space_dimension=space_dimension, random_state=random_state)
    manifold1._create_manifold()
    manifold2 = Line(space_dimension=space_dimension, random_state=3)
    manifold2._create_manifold()
    params = manifold1.get_params()
    manifold2.set_params(params)
    np.testing.assert_equal(manifold1.origin, manifold2.origin)
    np.testing.assert_equal(manifold1.direction, manifold2.direction)


@pytest.mark.parametrize("space_dimension, random_state", [(2, 1), (2, 2), (4, 10), (3, 1), (10, 1)])
def test_clone(space_dimension, random_state):
    manifold = Line(space_dimension=space_dimension, random_state=random_state)
    manifold._create_manifold()
    manifold_clone = manifold.clone()
    np.testing.assert_equal(manifold.origin, manifold_clone.origin)
    np.testing.assert_equal(manifold.direction, manifold_clone.direction)
