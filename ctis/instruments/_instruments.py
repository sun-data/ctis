from typing import Callable, Sequence
import abc
import dataclasses
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractInstrument",
    "IdealInstrument",
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
    An interface describing a general CTIS instrument.

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
    def backproject(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> ProjectionCallable:
        """
        The deprojection model of the CTIS instrument.
        Maps spectral and spatial coordinates on the detector to coordinates
        on the field.

        Parameters
        ----------
        images
            The number of electrons gathered by each pixel in every channel.
        """


@dataclasses.dataclass
class AbstractLinearInstrument(
    AbstractInstrument,
):
    """
    An instrument that can be modeled using matrix multiplication.
    """

    @property
    @abc.abstractmethod
    def _weights(self) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        A sparse matrix which maps spectral radiance on the skyplane to
        the number of electrons measured by the sensor.
        """

    def project(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        pass



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
