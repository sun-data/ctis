from typing import Callable
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractInstrument",
    "Instrument",
]


ProjectionCallable = Callable[
    [na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]],
    na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
]


@dataclasses.dataclass
class AbstractInstrument(
    abc.ABC,
):
    """
    An interface describing a CTIS instrument.

    This consists of a forward model
    (which maps spectral/spatial points on the skyplane to positions on the detector)
    and a deprojection model
    (which maps positions on the detector to spectral/spatial points on the skyplane).
    """

    @property
    @abc.abstractmethod
    def project(
        self,
    ) -> ProjectionCallable:
        """
        The forward model of the CTIS instrument.
        Maps spectral and spatial coordinates on the field to coordinates
        on the detector.
        """

    @property
    @abc.abstractmethod
    def deproject(
        self,
    ) -> ProjectionCallable:
        """
        The deprojection model of the CTIS instrument.
        Maps spectral and spatial coordinates on the detector to coordinates
        on the field.
        """


@dataclasses.dataclass
class Instrument(
    AbstractInstrument,
):
    """
    A CTIS instrument where the forward and deprojection models are explicitly
    provided.
    """

    project: ProjectionCallable = dataclasses.MISSING
    """
    The forward model of the CTIS instrument.
    Maps spectral and spatial coordinates on the field to coordinates
    on the detector.
    """

    deproject: ProjectionCallable = dataclasses.MISSING
    """
    The deprojection model of the CTIS instrument.
    Maps spectral and spatial coordinates on the detector to coordinates
    on the field.
    """
