from cblearn.datasets._base import BaseManifold
import numpy as np
from scipy.stats import ortho_group
from sklearn.utils import check_random_state
from typing import Union, Dict, Callable
from scipy.spatial.distance import pdist, squareform


class LinearSubspace(BaseManifold):
    """
    Linear Subspace

    Linear Subspace is a class for creating a hyperplane of a given
    subspace dimension embedded in a higher dimensional space. It gives
    a method of generating synthetic points with intrinsic
    structure and dimensionality. The generated points are then meant
    to be used for generating ordinal data.

    A reason for needing synthetically generated points is that it is
    often difficult to comprehensively evaluate the performance of
    ordinal methods on real data. Being able to modify the underlying
    geometry and structure of the data allows for better
    experimentation and control in evaluating ordinal methods.

    This class inherits from the BaseManifold class. This class creates
    hyperplanes reproducibly using the scipy.stats.ortho_group function
    for a given random state. This class can sample points from the
    hyperplane using a given sampling function and add noise to the
    points using a given noise function.

    .. note:: Subspace dimension must be less than or equal to space
    dimension and space dimension must be greater than 1.

    Attributes:
        subspace_dimension: Dimension of the subspace
        space_dimension: Dimension of the space
        random_state: Random state for reproducibility of the manifold
        created: Flag to check if the hyperplane has been created
        basis: Basis of the hyperplane

    Examples:
        >>> from cblearn.datasets import LinearSubspace, make_random_triplet_indices, triplet_response
        >>> # Creates a 1-dimensional hyperplane in 3-dimensional space
        >>> manifold = LinearSubspace(subspace_dimension=1, space_dimension=3)
        >>> # Samples 10 points from the created hyperplane
        >>> points, distances = manifold.sample_points(num_points=10)
        >>> print(points.shape)
        (10, 3)
        >>> print(distances.shape)
        (10, 10)
        >>> # Sampling 10 points with noise
        >>> noisy_points, noisy_distances = manifold.sample_points(10, noise='normal', noise_options={'scale': 0.1})
        >>> # Responding to triplets based on distance matrix
        >>> triplets = make_random_triplet_indices(n_objects=10, size=100)
        >>> response = triplet_response(triplets, distances, distance='precomputed')
    """
    def __init__(self, subspace_dimension: int, space_dimension: int,
                 random_state: Union[None, int, np.random.RandomState] = None):
        """
        Initialize the manifold

        Args:
            subspace_dimension: Dimension of the hyperplane
            space_dimension: Dimension of the space in which the hyperplane
                              is embedded
            random_state: The seed of the pseudo random number generator
                          to use when sampling. If None, the random number
                          generator is the RandomState instance used by
                          np.random.
        """
        if not isinstance(subspace_dimension, int):
            raise ValueError('Subspace dimension must be an integer')
        if subspace_dimension < 1:
            raise ValueError('Subspace dimension cannot be less than 1')
        if not isinstance(space_dimension, int):
            raise ValueError('Space dimension must be an integer')
        if subspace_dimension > space_dimension:
            raise ValueError('Subspace dimension cannot be greater than'
                             ' dimension')
        if space_dimension <= 1:
            raise ValueError('Space dimension cannot be less than 2')
        self.subspace_dimension = subspace_dimension
        self.space_dimension = space_dimension
        random_state = check_random_state(random_state)
        self.manifold_state = random_state
        self.created = False
        super().__init__(subspace_dimension=subspace_dimension,
                         space_dimension=space_dimension,
                         random_state=random_state)

    def _create_manifold(self):
        """ Creates the hyperplane """
        # Source:
        # https://stackoverflow.com/questions/69036765/sampling-random-points-from-linear-subspaces-of-a-given-radius-in-arbitary-dimen
        if self.subspace_dimension == 1:
            scipy_random_generator = ortho_group
            scipy_random_generator.random_state = self.manifold_state
            basis = scipy_random_generator.rvs(dim=self.space_dimension)[:2]
        else:
            scipy_random_generator = ortho_group
            scipy_random_generator.random_state = self.manifold_state
            basis = scipy_random_generator.rvs(dim=self.space_dimension)[
                :self.subspace_dimension]
        self.basis = basis
        self.created = True

    def sample_points(self, num_points: int,
                      sampling_function: Union[str, Callable] = 'normal',
                      sampling_options: Dict = {'scale': 1},
                      noise: Union[None, str, Callable] = None,
                      noise_options: Dict = {},
                      random_state: Union[None, int, np.random.RandomState] = None,
                      return_distances: bool = True):
        """
        Sample points from the hyperplane and add noise if requested

        Args:
            num_points: Number of points to sample
            sampling_function: The sampling function to use.
                               If a string, it should be a method of
                               the random state object. If a callable,
                               it should be a function that takes a
                               size argument and returns a numpy array
                               of samples.
            sampling_options: The options to pass to the sampling function.
            noise: The noise function to use. If a string, it should be
                   a method of the random state object. If a callable,
                   it should be a function that takes a size argument and
                   returns a numpy array of samples.
            noise_options: The options to pass to the noise function.
            random_state: The seed of the pseudo random number generator
                          to use when sampling. If None, the random number
                          generator is the RandomState instance used by
                          np.random.
            return_distances: Flag to return the distance matrix of
                              the sampled points. Defaults to True.

        Returns:
            The sampled points. If return_distances is True, the distance
            matrix of the sampled points (num_points, num_points) is also
            returned.
        """
        # Create Manifold if not already created
        if not self.created:
            self._create_manifold()

        # Get Noise Function
        if isinstance(noise, str):
            random_state = check_random_state(random_state)
            noise_fun: Callable = getattr(random_state, noise)
        elif callable(noise):
            noise_fun = noise

        # Get Sampling Function
        if isinstance(sampling_function, str):
            random_state = check_random_state(random_state)
            sampling_fun: Callable = getattr(random_state, sampling_function)
        elif callable(sampling_function):
            sampling_fun = sampling_function

        # Sample Coefficients
        if self.subspace_dimension == 1:
            coefficients = sampling_fun(
                size=(num_points, 1), **sampling_options)
            points = np.matmul(coefficients.reshape(-1, 1),
                               self.basis[0].reshape(1, -1)) + self.basis[1]
        else:
            coefficients = sampling_fun(
                size=(num_points, self.subspace_dimension),
                **sampling_options)
            points = np.matmul(coefficients, self.basis)

        # Add noise if requested
        if noise is not None:
            noise = noise_fun(size=points.shape, **noise_options)
            points = points + noise

        if return_distances:
            return points, self.get_canonical_distance_matrix(points)
        else:
            return points

    def get_canonical_distance_matrix(self, points: np.ndarray):
        """
        Get the distance matrix of the points sampled

        Args:
            points: The points sampled from the hyperplane

        Returns:
            The distance matrix of the points sampled (num_points,
            num_points)
        """
        return squareform(pdist(points))
