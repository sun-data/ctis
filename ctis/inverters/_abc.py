import abc
import dataclasses
import named_arrays as na
import numpy as np

__all__ = [
    "AbstractInverter",
]


@dataclasses.dataclass
class AbstractInverter(
    abc.ABC,
):
    """
    An interface describing an algorithm which can invert CTIS observations
    to yield a reconstruction of the observed scene.
    """
    @property
    @abc.abstractmethod
    def instrument(self) -> na.ScalarArray(np.ndarray[list[tuple[float, float, float]]]):
        """
        tuple returned by weights (shape_input, shape_output, weights)
        Returns
        -------

        """


    def __call__(
        self,
        data: na.FunctionArray,
        **kwargs,
    )-> ctis.results.AbstractResults:
        """
        Invert the given image deprojections to reconstruct the original scene.

        Parameters
        ----------
        data
            Collection of images to which the inverted solution is compared.

        kwargs
            Additional keyword arguments which can be used by subclass
            implementations.
        """

