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
    def weights(self) -> na.ScalarArray(np.ndarray[list[tuple[float, float, float]]]):
        """
        tuple returned by weights (shape_input, shape_output, weights)
        Returns
        -------

        """

    @property
    @abs.abstractmethod
    def weights_backward(self) -> np.ndarray[list[tuple[float, float, float]]]:
        return


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

    @abc.abstractmethod
    def project(self):
        """

        Returns
        -------

        """

    @abc.abstractmethod
    def back_project(self):
        """

        Returns
        -------

        """