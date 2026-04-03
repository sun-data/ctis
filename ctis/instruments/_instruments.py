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

    The most important method of this interface is :meth:`image`,
    which represents the forward model of the instrument and maps the
    spectral radiance of the skyplane to detector counts.

    The other important method of this interface is :meth:`deproject`,
    which is the transpose of image and maps detector counts from an
    observed image to the corresponding spectral radiance on the skyplane.
    """

    @abc.abstractmethod
    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        r"""
        The forward model of this CTIS instrument, which maps spectral radiance
        on the skyplane to counts on the detectors.
        
        Parameters
        ----------
        scene
            The spectral radiance in units equivalent to 
            erg / cm^2 / sr^2 / nm / s.
            :math:`\unit{\erg\per\cm\squared\per\steradian\per\nm\per\second}`
            {(u.erg / (u.cm**2 * u.sr * u.AA * u.s)):latex_inline}.
        """

    @abc.abstractmethod
    def deproject(
        self,
        image: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        A quasi-inverse model of this CTIS instrument, which maps counts
        on the detectors to spectral radiance on the skyplane.

        This is not a true inverse, since this just spreads intensity out
        evenly along each projection direction, and doesn't concentrate it
        in its true location.

        Parameters
        ----------
        image
            A series of images captured by a CTIS instrument.
            Should be in units of electrons.
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

    @property
    @abc.abstractmethod
    def coordinates_sensor(self) -> na.AbstractSpectralPositionalVectorArray:
        """
        A grid of wavelength and position coordinates on the detector plane.
        """


@dataclasses.dataclass
class AbstractLinearInstrument(
    AbstractInstrument,
):
    """
    An instrument where the forward model can be represented using
    matrix multiplication.
    """

    @property
    @abc.abstractmethod
    def weights(self) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        The contribution of each voxel on the skyplane to each pixel on the
        detector.
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

    response: u.Quantity | na.AbstractScalar
    """The number of electrons measured for a given spectral radiance on the skyplane."""

    plate_scale: u.Quantity | na.AbstractScalar
    r"""The spatial scale of the image on the sensor in :math:`\text{arcsec} \,\text{pix}^-1`"""

    dispersion: u.Quantity | na.AbstractScalar
    r"""The magnitude of the dispersion in :math:`\text{m \AA} \,\text{pix}^-1`"""

    angle: u.Quantity | na.AbstractScalar
    """The angle of the dispersion direction with respect to the scene."""
