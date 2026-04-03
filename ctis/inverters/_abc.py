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
    def instrument(self) -> ctis.instrument.AbstractInstrument:
        """
        An instrument maps data in coordinates_scene to coordinates_detector, and possibly back, using project and backproject
        methods.
        """

    def __call__(
        self,
        data: na.FunctionArray,
        **kwargs,
    )-> ctis.results.AbstractResults:
        """
        Calculate a reconstruction of a scene (observered by instrument) that is consistent with observed data.

        Parameters
        ----------
        data
            Collection of images to which the inverted solution is compared.

        kwargs
            Additional keyword arguments which can be used by subclass
            implementations.
        """

