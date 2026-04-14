import abc
import dataclasses
import named_arrays as na
import ctis
from ._results import InversionResult

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
    def instrument(self) -> ctis.instruments.AbstractInstrument:
        """
        A model of a CTIS instrument which transforms the radiance of an observed
        scene to photons measured by the sensors.
        """

    @abc.abstractmethod
    def __call__(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        **kwargs,
    )-> InversionResult:
        """
        Reconstruct a scene using the observed images.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
            Must be evaluated on the same coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_sensor`
            attribute of :attr:`instrument`.
        kwargs
            Additional keyword arguments which can be used by subclass
            implementations.
        """