import warnings
import dataclasses
import numpy as np
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
    An inversion routine based on the multiplicative algebraic reconstruction
    technique (MART) :cite:t:`Gordon1970`.

    For further information, see the discussion :doc:`../discussions/mart-discussion`.
    """

    instrument: ctis.instruments.AbstractInstrument = dataclasses.MISSING
    """
    A model of a CTIS instrument which transforms the radiance of an observed
    scene to photons measured by the sensors.
    """

    gamma: None | float = None
    r"""
    Learning rate, :math:`\gamma`.
    
    At every iteration, the current correction, :math:`C`, is replaced by 
    :math:`C^\gamma`.
    
    If :obj:`None`, :math:`\gamma = 2 / N`, where :math:`N` is the number of
    channels.
    """

    threshold_convergence: float = 1e-3
    r"""
    The convergence threshold, :math:`T`, which halts the iteration.
    
    If :math:`\langle \chi_{i-1}^2 \rangle - \langle \chi_{i}^2 \rangle < T`,
    then the algorithm is considered to be converged.
    """

    def __post_init__(self):

        if self.gamma is None:
            self.gamma = 2 / self.instrument.num_channel

    def __call__(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.ScalarArray = None,
        verbose: bool = False,
    ) -> IterativeInversionResult:
        """
        Reconstruct a scene using the observed images.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
            Must be evaluated on the same position coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_sensor`
            attribute of :attr:`instrument`.
        guess
            The initial guess at the reconstructed scene.
            Must be evaluated on the same coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_scene`
            attribute of :attr:`instrument`.
        """

        instrument = self.instrument

        axis_channel = instrument.axis_channel

        position_images = images.inputs.position
        position_sensor = instrument.coordinates_sensor.position
        if not np.all(position_images == position_sensor):
            raise ValueError(
                "`images.inputs.position` and `self.coordinates_sensor.position` "
                "are not equal."
            )
        images_inputs = images.inputs
        images = images.outputs

        if guess is None:
            scene = instrument.backproject(images).outputs
            scene = scene.mean(axis_channel)
            scene.ndarray[:] = scene.ndarray.mean()
        else:
            scene = guess.copy()

        num_channel = instrument.num_channel

        gamma = self.gamma

        backprojected = instrument.backproject(images).outputs

        backprojected = np.maximum(backprojected, 0)

        intermediate = []

        merit_old = np.inf

        chi2 = []
        correlation_residual = []

        for i in range(self.num_iteration):

            if self.intermediate:
                intermediate.append(scene)

            if verbose:  # pragma: nocover
                print(f"{i=}")

            images_new = instrument.image(scene, noise=False).outputs

            chi2_ij = self.mean_chi_squared(images, images_new)
            r_ij = self.correlation_residual(images, images_new)

            chi2.append(chi2_ij)
            correlation_residual.append(r_ij)

            merit = chi2_ij.mean(axis_channel)

            if verbose:  # pragma: nocover
                print(f"merit: {merit}")

            elif (merit_old - merit) < self.threshold_convergence:
                message = f"Achieved merit less than {self.threshold_convergence}."
                success = True
                num_iteration = i + 1
                break

            backprojected_new = instrument.backproject(images_new).outputs

            backprojected_new = np.maximum(backprojected_new, 0)

            correction = backprojected / backprojected_new

            correction = np.nan_to_num(
                x=correction,
                nan=1,
                posinf=1,
                neginf=1,
            )

            correction = correction**gamma

            correction = np.prod(correction, axis=instrument.axis_channel)
            correction = correction ** (1 / num_channel)

            if self.intermediate:
                scene = scene * correction
            else:
                scene *= correction

            merit_old = merit

        else:
            message = f"Max number of iterations ({self.num_iteration}) exceeded."
            warnings.warn(message)
            success = False
            num_iteration = self.num_iteration

        if self.intermediate:
            intermediate = na.stack(intermediate, axis=self.axis_iteration)
            solutions = intermediate
        else:
            solutions = scene.add_axes(self.axis_iteration)

        solutions = na.FunctionArray(
            inputs=self.instrument.coordinates_scene,
            outputs=solutions,
        )

        images = na.FunctionArray(
            inputs=images_inputs,
            outputs=images,
        )

        mean_chi_squared = na.stack(chi2, axis=self.axis_iteration)
        correlation_residual = na.stack(correlation_residual, axis=self.axis_iteration)

        return IterativeInversionResult(
            solutions=solutions,
            success=success,
            images=images,
            inverter=self,
            message=message,
            num_iteration=num_iteration,
            mean_chi_squared=mean_chi_squared,
            correlation_residual=correlation_residual,
        )
