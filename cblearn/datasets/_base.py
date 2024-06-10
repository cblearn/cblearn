from abc import ABC, abstractmethod
from copy import deepcopy


class BaseManifold(ABC):
    """
    Base class for manifold samplers.
    """

    def __init__(self, **kwargs):
        """
        Initialize the manifold
        """
        self._params = {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def _create_manifold(self):
        """
        Create the manifold
        """
        raise NotImplementedError

    @abstractmethod
    def sample_points(self, num_points: int, **kwargs):
        """
        Sample points from the manifold
        """
        raise NotImplementedError

    @abstractmethod
    def sample_triplets(self, num_points: int, num_triplets: int, **kwargs):
        """
        Sample triplets from the manifold directly
        """
        raise NotImplementedError

    @abstractmethod
    def get_canonical_distance_matrix(self, **kwargs):
        """
        Get the distance matrix of the points sampled
        """
        raise NotImplementedError

    def get_params(self):
        """
        Get the parameters of the manifold

        Returns:
            The parameters of the manifold
        """
        return {
            attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_")
        }

    def set_params(self, params):
        """
        Set the parameters of the manifold

        Args:
            The parameters to set
        """
        for attr, value in params.items():
            setattr(self, attr, value)

    def clone(self):
        """
        Clone the manifold

        Returns:
            A clone of the manifold
        """
        return deepcopy(self)
