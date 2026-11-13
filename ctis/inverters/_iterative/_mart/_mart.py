import warnings
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from ctis.instruments._operators import Operators, aligned
from .. import AbstractIterativeInverter, IterativeInversionResult
from . import _kernels

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

    If the instrument can be expressed as a pair of sparse matrices, as an
    :class:`~ctis.instruments.IdealInstrument` can, the iteration runs on
    those, using plain arrays and compiled code.
    Otherwise, it runs on the :meth:`~ctis.instruments.AbstractInstrument.image`
    and :meth:`~ctis.instruments.AbstractInstrument.backproject` methods of
    the instrument, which is slower.
    The two give the same result.
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

        position_images = images.inputs.position
        position_sensor = instrument.coordinates_sensor.position
        if not np.all(position_images == position_sensor):
            raise ValueError(
                "`images.inputs.position` and `self.coordinates_sensor.position` "
                "are not equal."
            )

        arrays = self._arrays(images.outputs, guess)

        if arrays is None:
            return self._call_named(images, guess, verbose)

        return self._call_arrays(images, guess, verbose, *arrays)

    def _arrays(
        self,
        images: na.AbstractScalar,
        guess: None | na.AbstractScalar,
    ) -> None | tuple[Operators, np.ndarray, None | np.ndarray]:
        """
        The operators of the instrument, with the observed images and the
        initial guess as the plain arrays that the operators act on, or
        :obj:`None` if any of them cannot be expressed that way.

        The guess is a copy, which the iteration is free to overwrite, or
        :obj:`None` if there is none.

        Parameters
        ----------
        images
            The values of the observed images.
        guess
            The initial guess at the reconstructed scene, if any.
        """
        operators = self.instrument._operators
        if operators is None:
            return None

        shape_images = operators.shape_images
        shape_scene = operators.shape_scene

        observed = aligned(images, shape=shape_images, unit=operators.unit_images)
        if observed is None:
            return None
        observed = np.broadcast_to(observed, tuple(shape_images.values()))
        observed = np.ascontiguousarray(observed)

        scene = None
        if guess is not None:
            scene = aligned(guess, shape=shape_scene, unit=operators.unit_scene)
            # a guess without every axis of the scene is left to the named
            # arrays, which do not accept one
            if scene is None or scene.shape != tuple(shape_scene.values()):
                return None
            scene = np.array(scene, order="C")

        return operators, observed, scene

    def _call_arrays(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.ScalarArray,
        verbose: bool,
        operators: Operators,
        observed: np.ndarray,
        scene: None | np.ndarray,
    ) -> IterativeInversionResult:
        """
        Reconstruct a scene by iterating on the operators of the instrument,
        using plain arrays and compiled code.

        This is the same algorithm as :meth:`_call_named`, and it gives the
        same result.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
        guess
            The initial guess at the reconstructed scene.
        verbose
            Whether to print the progress of the iteration.
        operators
            The operators of the instrument.
        observed
            The values of `images`, as the operators take them.
        scene
            The values of `guess`, as the operators take them, which can be
            overwritten, or :obj:`None` if there is no guess.
        """
        num_channel = operators.num_channel

        backprojected = operators.backproject(observed)

        if scene is None:
            # the mean of the backprojection of the instrument, in every voxel
            mean = backprojected * operators.scale_backproject
            mean = mean.reshape(num_channel, -1).mean(axis=0).mean()
            scene = np.full(tuple(operators.shape_scene.values()), mean)

        # the voxels of the scene, which each iteration overwrites, and
        # which the scene is a view of whatever its layout was
        voxels = np.ascontiguousarray(scene).reshape(-1)
        scene = voxels.reshape(scene.shape)

        observed = observed.reshape(num_channel, -1)
        backprojected = backprojected.reshape(num_channel, -1)

        intermediate = []

        merit_old = np.inf

        chi2 = np.empty((self.num_iteration, num_channel))
        correlation_residual = np.empty((self.num_iteration, num_channel))

        for i in range(self.num_iteration):

            if self.intermediate:
                intermediate.append(scene.copy())

            if verbose:  # pragma: nocover
                print(f"{i=}")

            predicted, variance = operators.image(scene)

            _kernels.merit(
                observed=observed,
                predicted=predicted.reshape(num_channel, -1),
                variance=variance.reshape(num_channel, -1),
                chi_squared=chi2[i],
                correlation=correlation_residual[i],
            )

            merit = chi2[i].mean()

            if verbose:  # pragma: nocover
                print(f"merit: {merit}")

            if (merit_old - merit) < self.threshold_convergence:
                message = f"Achieved merit less than {self.threshold_convergence}."
                success = True
                num_iteration = i + 1
                break

            backprojected_new = operators.backproject(predicted)

            _kernels.update(
                scene=voxels,
                backprojected=backprojected,
                backprojected_new=backprojected_new.reshape(num_channel, -1),
                gamma=self.gamma,
            )

            merit_old = merit

        else:
            message = f"Max number of iterations ({self.num_iteration}) exceeded."
            warnings.warn(message)
            success = False
            num_iteration = self.num_iteration

        if self.intermediate:
            solutions = np.stack(intermediate)
        else:
            solutions = scene[np.newaxis]

        # the solution is in the units of the guess, if there is one
        solutions = solutions << operators.unit_scene
        if guess is not None:
            solutions = solutions.to(guess.unit)

        shape_channel = tuple(operators.shape_channel.values())
        shape_merit = (num_iteration,) + shape_channel
        chi2 = chi2[:num_iteration].reshape(shape_merit)
        correlation_residual = correlation_residual[:num_iteration].reshape(shape_merit)

        axes_merit = (self.axis_iteration, *operators.shape_channel)

        return IterativeInversionResult(
            solutions=na.FunctionArray(
                inputs=self.instrument.coordinates_scene,
                outputs=na.ScalarArray(
                    ndarray=solutions,
                    axes=(self.axis_iteration, *operators.shape_scene),
                ),
            ),
            success=success,
            images=images,
            inverter=self,
            message=message,
            num_iteration=num_iteration,
            mean_chi_squared=na.ScalarArray(
                ndarray=chi2 << u.dimensionless_unscaled,
                axes=axes_merit,
            ),
            correlation_residual=na.ScalarArray(
                ndarray=correlation_residual << u.dimensionless_unscaled,
                axes=axes_merit,
            ),
        )

    def _call_named(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.ScalarArray,
        verbose: bool,
    ) -> IterativeInversionResult:
        """
        Reconstruct a scene by iterating on the
        :meth:`~ctis.instruments.AbstractInstrument.image` and
        :meth:`~ctis.instruments.AbstractInstrument.backproject` methods of
        the instrument, using named arrays.

        This works for any instrument, and it is what :meth:`_call_arrays`
        is tested against.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
        guess
            The initial guess at the reconstructed scene.
        verbose
            Whether to print the progress of the iteration.
        """

        instrument = self.instrument

        axis_channel = instrument.axis_channel

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

            predicted = instrument.image(scene, noise=False, uncertainty=True).outputs
            images_new = predicted.nominal

            chi2_ij = self.mean_chi_squared(images, images_new, predicted.width)
            r_ij = self.correlation_residual(images, images_new)

            chi2.append(chi2_ij)
            correlation_residual.append(r_ij)

            merit = chi2_ij.mean(axis_channel)

            if verbose:  # pragma: nocover
                print(f"merit: {merit}")

            if (merit_old - merit) < self.threshold_convergence:
                message = f"Achieved merit less than {self.threshold_convergence}."
                success = True
                num_iteration = i + 1
                break

            backprojected_new = instrument.backproject(images_new).outputs

            backprojected_new = np.maximum(backprojected_new, 0)

            # voxels outside the field of view of a channel receive no signal,
            # and their undefined corrections are replaced with unity below.
            with np.errstate(divide="ignore", invalid="ignore"):
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
