from typing import Callable, Sequence
import abc
import dataclasses
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractInstrument",
    "AbstractLinearInstrument",
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

    The only member of this interface is :meth:`image`,
    which represents the forward model of the instrument.

    This consists of a forward model
    (which maps the spectral radiance of a physical scene to counts on a detector)
    and a deprojection model
    (which maps detector counts to the spectral radiance of a physical scene).
    """

    @abc.abstractmethod
    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        f"""
        The forward model of this CTIS instrument, which maps spectral radiance
        on the skyplane to counts on the detectors.
        
        Parameters
        ----------
        scene
            The spectral radiance in units equivalent to 
            {(u.erg / (u.cm**2 * u.sr * u.AA * u.s)):latex_inline}.
        """

    @property
    @abc.abstractmethod
    def coordinates_scene(self) -> na.AbstractSpectralPositionalVectorArray:
        """
        A grid of wavelength and position coordinates on the skyplane
        which will be used to construct the inverted scene.

        Normally the pitch of this grid is chosen to be the average
        plate scale of the instrument.
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

    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        pass

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
