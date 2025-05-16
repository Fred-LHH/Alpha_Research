import abc


class BaseOptimizer(abc.ABC):

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> object:
        """Generate a optimized portfolio allocation"""
