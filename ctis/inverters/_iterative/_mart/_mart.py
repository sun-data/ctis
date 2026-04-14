from typing import ClassVar
import warnings
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from .. import AbstractIterativeInverter, IterativeInversionResult

__all__ = [
    "MartInverter",
]

@dataclasses.dataclass
class MartInverter(
    AbstractIterativeInverter,
):
    """
    An inversion routine based on the Richardson-Lucy algorithm
    :cite:t:`Richardson1972,Lucy1974`.
    """

    instrument: ctis.instruments.AbstractInstrument = dataclasses.MISSING
    """
    A model of a CTIS instrument which transforms the radiance of an observed
    scene to photons measured by the sensors.
    """

    gamma: None | float = None
    r"""
    Contrast-enhancement factor, :math:`\gamma`.
    
    At every iteration, the current guess, :math:`G`, is replaced by 
    :math:`G^\gamma`.
    
    If :obj:`None`, :math:`\gamma = 2 / N`, where :math:`N` is the number of
    channels.
    """

    num_iteration: int = 100
    """
    The maximum number of iterations to perform.
    
    If convergence is not reached before this number is exceeded,
    a warning is raised and an unsuccessful result is returned. 
    """

    intermediate: bool = False
    """
    Whether to save intermediate solutions.
    
    This is set to :obj:`False` during normal operation, but can be useful for
    debugging or demonstration purposes.
    """

    @property
    def _gamma(self) -> float:
        """Normalized version of :attr:`gamma`"""
        gamma = self.gamma
        if gamma is None:
            gamma = 2 / self.instrument.num_channel
        return gamma

    def __call__(
        self,
        images: na.ScalarArray,
        guess: None | na.ScalarArray = None,
    ) -> IterativeInversionResult:
        """
        Reconstruct a scene using the observed images.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
            Must be evaluated on the same coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_sensor`
            attribute of :attr:`instrument`.
        guess
            The initial guess at the reconstructed scene.
            Must be evaluated on the same coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_scene`
            attribute of :attr:`instrument`.
        """

        scene = guess.copy()

        instrument = self.instrument

        num_channel = instrument.num_channel

        gamma = self._gamma

        backprojected = instrument.backproject(images).outputs

        intermediate = []
        if self.intermediate:
            intermediate.append(scene)

        chi_squared = self._mean_chi_squared(images, 0 * images.unit)

        merit = []

        for i in range(self.num_iteration):

            images_new = instrument.image(scene, noise=False).outputs

            chi_squared_new = self._mean_chi_squared(images, images_new)

            if (chi_squared - chi_squared_new) < 1e-2:
            # if self._converged(images, images_new):
                message = "Achieved mean chi squared of less than 1."
                success = True
                break

            backprojected_new = instrument.backproject(images_new).outputs

            correction = backprojected / backprojected_new

            correction = np.nan_to_num(
                x=correction,
                nan=1,
                posinf=1,
                neginf=1,
            )

            correction = correction ** gamma

            correction = np.prod(correction, axis=instrument.axis_channel)

            correction = correction ** (1 / num_channel)

            if self.intermediate:
                scene = scene * correction
                intermediate.append(scene)

            else:
                scene *= correction

            merit.append(chi_squared_new)

            chi_squared = chi_squared_new

        else:
            message = f"Max number of iterations ({self.num_iteration}) exceeded."
            warnings.warn(message)
            success = False

        if self.intermediate:
            intermediate = na.stack(intermediate, axis=self.axis_iteration)
            solution = intermediate
        else:
            solution = scene

        return IterativeInversionResult(
            solution=solution,
            success=success,
            images=images,
            inverter=self,
            message=message,
            num_iteration=i,
            merit=na.stack(merit, axis=self.axis_iteration),
            merit_name=r"$\langle \chi^2 \rangle$",
        )

    def _converged(
        self,
        images: na.ScalarArray,
        images_new: na.ScalarArray,
    ) -> bool:
        r"""
        Return true if :math:`\langle \chi^2 \rangle < 1`
        """
        X2 = self._mean_chi_squared(images, images_new)
        print(f"{X2=}")
        return X2 < 1/2

    def _mean_chi_squared(
        self,
        images: na.ScalarArray,
        images_new: na.ScalarArray,
    ):
        r"""
        Evaluated :math:`\langle \chi^2 \rangle < 1` normalized by uncertainty in each pixel.
        """

        uncertainty = self.instrument.uncertainty(images_new)

        uncertainty = np.maximum(uncertainty, 1  * u.photon)

        return np.mean(np.square((images_new - images) / uncertainty))
