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
    weights_forward: na.ScalarArray(np.ndarray[list[tuple[float, float, float]]])
    shape_solution: dict
    weights_backward: np.ndarray[list[tuple[float, float, float]]] = None

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

    def merit(self):
        """
        Something to calculate inversion accuracy
        Returns
        -------
        float
        """