from typing import Callable, Sequence
import abc
import dataclasses
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractInstrument",
    "IdealInstrument",
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
    (which maps the spectral radiance of a physical scene to counts on a detector)
    and a deprojection model
    (which maps detector counts to the spectral radiance of a physical scene).
    """

    @abc.abstractmethod
    def project(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The forward model of the CTIS instrument.
        Maps spectral and spatial coordinates on the field to coordinates
        on the detector.

        Parameters
        ----------
        scene
            The spectral radiance of each spatial/spectral point in the scene.
        """

    @abc.abstractmethod
    def deproject(
        self,
        projections: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> ProjectionCallable:
        """
        The deprojection model of the CTIS instrument.
        Maps spectral and spatial coordinates on the detector to coordinates
        on the field.

        Parameters
        ----------
        projections
            The counts gathered by each detector in the CTIS instrument.
        """


@dataclasses.dataclass
class IdealInstrument(
    AbstractInstrument,
):
    """
    An idealized CTIS instrument which has a perfect point-spread function
    and no noise.
    """

    dispersion: u.Quantity | na.AbstractScalar
    r"""The magnitude of the dispersion in :math:`\text{m \AA} \,\text{pix}^-1`"""

    angle: u.Quantity | na.AbstractScalar
    """The angle of the dispersion direction with respect to the scene."""


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
