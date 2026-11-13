import math
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

    Several frames, sets of images of the same instrument, can be
    reconstructed in one call, see :meth:`__call__`.
    On the matrices, the frames are iterated on together and share each pass
    over them, which takes less time for each frame than a pass of its own.
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

    dtype: type = dataclasses.field(default=np.float64, kw_only=True)
    """
    The precision of the iteration,
    :obj:`numpy.float64` or :obj:`numpy.float32`.

    In single precision, the scenes, the images and the sparse matrices of
    the instrument take half the memory, which the largest problems need,
    and the solution is given in single precision.
    Every sum is still accumulated in double precision, and the merit of
    each iteration is given in double precision.

    Single precision is only available for an instrument which can be
    expressed as a pair of sparse matrices.
    """

    def __post_init__(self):

        if self.gamma is None:
            self.gamma = 2 / self.instrument.num_channel

        if np.dtype(self.dtype) not in (np.float32, np.float64):
            raise ValueError(
                f"`dtype` must be `numpy.float32` or `numpy.float64`, "
                f"got {self.dtype}."
            )

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
            Any axes beyond those of the channels and the sensor are those of
            independent frames, sets of images of the same instrument.
            The scene of each frame is reconstructed on its own, and stops
            at its own iteration, exactly as it would alone.
        guess
            The initial guess at the reconstructed scene, the same for every
            frame, or one for each frame.
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

        shape_frames = self._shape_frames(images.outputs)

        arrays = self._arrays(images.outputs, guess, shape_frames)

        if arrays is not None:
            return self._call_arrays(images, guess, verbose, shape_frames, *arrays)

        # the methods of an instrument compute in the precision of its own
        # arrays, whatever precision is asked for here
        dtype = np.dtype(self.dtype)
        if dtype != np.float64 and instrument._operators(dtype) is None:
            raise ValueError(
                f"`dtype` is {dtype}, but this instrument cannot be expressed as "
                f"a pair of sparse matrices, so it is inverted through its "
                f"`image` and `backproject` methods, in double precision."
            )

        if shape_frames:
            return self._call_frames(images, guess, verbose, shape_frames)

        return self._call_named(images, guess, verbose)

    def _shape_frames(self, images: na.AbstractScalar) -> dict[str, int]:
        """
        The axes of the frames and their lengths: the axes of the images
        beyond those of the channels and the sensor.

        Parameters
        ----------
        images
            The values of the observed images.
        """
        instrument = self.instrument

        axis_channel = instrument.axis_channel
        if isinstance(axis_channel, str):
            axis_channel = (axis_channel,)
        axis_images = (*axis_channel, *instrument.axis_sensor_xy)

        shape = na.shape(images)

        return {ax: shape[ax] for ax in shape if ax not in axis_images}

    def _arrays(
        self,
        images: na.AbstractScalar,
        guess: None | na.AbstractScalar,
        shape_frames: dict[str, int],
    ) -> None | tuple[Operators, np.ndarray, None | np.ndarray]:
        """
        The operators of the instrument, with the observed images and the
        initial guess as the plain arrays that the operators act on, or
        :obj:`None` if any of them cannot be expressed that way.

        Each array has the frames along one last axis.
        The guess is a copy, which the iteration is free to overwrite, or
        :obj:`None` if there is none.

        Parameters
        ----------
        images
            The values of the observed images.
        guess
            The initial guess at the reconstructed scene, if any.
        shape_frames
            The axes of the frames and their lengths.
        """
        operators = self.instrument._operators(self.dtype)
        if operators is None:
            return None

        num_frame = math.prod(shape_frames.values())

        shape_images = tuple(operators.shape_images.values())
        shape_scene = tuple(operators.shape_scene.values())

        observed = aligned(
            a=images,
            shape=operators.shape_images | shape_frames,
            unit=operators.unit_images,
        )
        if observed is None:
            return None
        observed = np.broadcast_to(
            observed, shape_images + tuple(shape_frames.values())
        )
        observed = np.ascontiguousarray(observed, dtype=operators.dtype)
        observed = observed.reshape(shape_images + (num_frame,))

        scene = None
        if guess is not None:
            scene = aligned(
                a=guess,
                shape=operators.shape_scene | shape_frames,
                unit=operators.unit_scene,
            )
            # a guess without every axis of the scene is left to the named
            # arrays, which do not accept one
            if scene is None or scene.shape[: len(shape_scene)] != shape_scene:
                return None
            scene = np.broadcast_to(scene, shape_scene + tuple(shape_frames.values()))
            scene = np.array(scene, dtype=operators.dtype, order="C")
            scene = scene.reshape(shape_scene + (num_frame,))

        return operators, observed, scene

    def _call_arrays(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.ScalarArray,
        verbose: bool,
        shape_frames: dict[str, int],
        operators: Operators,
        observed: np.ndarray,
        scene: None | np.ndarray,
    ) -> IterativeInversionResult:
        """
        Reconstruct the scene of every frame by iterating on the operators
        of the instrument, using plain arrays and compiled code.

        This is the same algorithm as :meth:`_call_named`, and it gives each
        frame the same result.
        The frames are iterated on together, so that they share each pass
        over the matrices, and a frame which has converged is left out of
        the passes which follow.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
        guess
            The initial guess at the reconstructed scene.
        verbose
            Whether to print the progress of the iteration.
        shape_frames
            The axes of the frames and their lengths.
        operators
            The operators of the instrument.
        observed
            The values of `images`, as the operators take them, with the
            frames along the last axis.
        scene
            The values of `guess`, as the operators take them, which can be
            overwritten, or :obj:`None` if there is no guess.
        """
        dtype = operators.dtype

        num_channel = operators.num_channel
        num_frame = observed.shape[~0]

        shape_scene = tuple(operators.shape_scene.values())
        num_voxel = math.prod(shape_scene)

        backprojected = operators.backproject(observed)

        if scene is None:
            # the mean of the backprojection of the instrument, in every
            # voxel, for each frame
            scene = np.empty((num_voxel, num_frame), dtype=dtype)
            for f in range(num_frame):
                mean = backprojected[..., f] * operators.scale_backproject[..., 0]
                mean = mean.reshape(num_channel, -1).mean(axis=0, dtype=np.float64)
                scene[:, f] = mean.mean()

        # the arrays of the frames which are still iterating, which every
        # iteration overwrites the scenes of
        scene = np.ascontiguousarray(scene).reshape(num_voxel, num_frame)
        observed = observed.reshape(num_channel, -1, num_frame)
        backprojected = backprojected.reshape(num_channel, num_voxel, num_frame)
        merit_old = np.full(num_frame, np.inf)

        # which frames those are
        active = np.arange(num_frame)

        solution = np.empty((num_voxel, num_frame), dtype=dtype)
        num_iteration = np.full(num_frame, self.num_iteration)
        success = np.zeros(num_frame, dtype=bool)

        # a frame has no merit once it has stopped
        shape_merit = (self.num_iteration, num_frame, num_channel)
        chi2 = np.full(shape_merit, np.nan)
        correlation_residual = np.full(shape_merit, np.nan)

        intermediate = []

        for i in range(self.num_iteration):

            if self.intermediate:
                # a frame which has stopped keeps its solution
                solution[:, active] = scene
                intermediate.append(solution.copy())

            if verbose:  # pragma: nocover
                print(f"{i=}")

            predicted, variance = operators.image(
                scene=scene.reshape(shape_scene + (active.size,)),
            )
            predicted = predicted.reshape(num_channel, -1, active.size)

            chi2_i = np.empty((active.size, num_channel))
            correlation_residual_i = np.empty((active.size, num_channel))
            _kernels.merit(
                observed=observed,
                predicted=predicted,
                variance=variance.reshape(num_channel, -1, active.size),
                chi_squared=chi2_i,
                correlation=correlation_residual_i,
            )
            chi2[i, active] = chi2_i
            correlation_residual[i, active] = correlation_residual_i

            merit = chi2_i.mean(axis=1)

            if verbose:  # pragma: nocover
                print(f"merit: {merit}")

            converged = (merit_old - merit) < self.threshold_convergence

            if np.any(converged):
                # the frames which converged stop here, before the update,
                # and the others carry on without them
                solution[:, active[converged]] = scene[:, converged]
                num_iteration[active[converged]] = i + 1
                success[active[converged]] = True

                active = active[~converged]
                if active.size == 0:
                    break

                scene = np.ascontiguousarray(scene[:, ~converged])
                observed = np.ascontiguousarray(observed[:, :, ~converged])
                backprojected = np.ascontiguousarray(backprojected[:, :, ~converged])
                predicted = np.ascontiguousarray(predicted[:, :, ~converged])
                merit = merit[~converged]

            backprojected_new = operators.backproject(predicted)

            _kernels.update(
                scene=scene,
                backprojected=backprojected,
                backprojected_new=backprojected_new.reshape(num_channel, num_voxel, -1),
                gamma=self.gamma,
            )

            merit_old = merit

        else:
            warnings.warn(f"Max number of iterations ({self.num_iteration}) exceeded.")
            solution[:, active] = scene

        if self.intermediate:
            solutions = np.stack(intermediate)
        else:
            solutions = solution[np.newaxis]
        solutions = solutions.reshape(
            (solutions.shape[0],) + shape_scene + (num_frame,)
        )

        # the solution is in the units of the guess, if there is one
        solutions = solutions << operators.unit_scene
        if guess is not None:
            solutions = solutions.to(guess.unit)

        return self._result(
            images=images,
            solutions=solutions,
            mean_chi_squared=chi2,
            correlation_residual=correlation_residual,
            num_iteration=num_iteration,
            success=success,
            shape_scene=operators.shape_scene,
            shape_channel=operators.shape_channel,
            shape_frames=shape_frames,
        )

    def _call_frames(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.ScalarArray,
        verbose: bool,
        shape_frames: dict[str, int],
    ) -> IterativeInversionResult:
        """
        Reconstruct the scene of every frame with :meth:`_call_named`, one
        frame at a time.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
        guess
            The initial guess at the reconstructed scene.
        verbose
            Whether to print the progress of the iteration.
        shape_frames
            The axes of the frames and their lengths.
        """
        axis_iteration = self.axis_iteration

        results = [
            self._call_named(
                images=images[index],
                guess=None if guess is None else guess[index],
                verbose=verbose,
            )
            for index in na.ndindex(shape_frames)
        ]

        solutions = [r.solutions.outputs for r in results]
        chi2 = [r.mean_chi_squared for r in results]
        correlation_residual = [r.correlation_residual for r in results]

        shape_scene = solutions[0].shape
        shape_scene = {
            ax: shape_scene[ax] for ax in shape_scene if ax != axis_iteration
        }
        shape_channel = chi2[0].shape
        shape_channel = {
            ax: shape_channel[ax] for ax in shape_channel if ax != axis_iteration
        }

        unit = solutions[0].unit
        axes_solutions = (axis_iteration, *shape_scene)
        axes_merit = (axis_iteration, *shape_channel)

        num_frame = len(results)
        num_channel = math.prod(shape_channel.values())
        num_iteration = np.array([r.num_iteration for r in results])
        num_solutions = max(s.shape[axis_iteration] for s in solutions)

        result_solutions = np.empty(
            shape=(num_solutions, *shape_scene.values(), num_frame),
        )
        shape_merit = (num_iteration.max(), num_frame, num_channel)
        result_chi2 = np.full(shape_merit, np.nan)
        result_correlation_residual = np.full(shape_merit, np.nan)

        for f in range(num_frame):
            s = solutions[f].ndarray_aligned(axes_solutions).to_value(unit)
            n = num_iteration[f]

            # a frame which has stopped keeps its solution, and has no merit
            result_solutions[: len(s), ..., f] = s
            result_solutions[len(s) :, ..., f] = s[~0]
            result_chi2[:n, f] = np.asarray(
                chi2[f].ndarray_aligned(axes_merit),
            ).reshape(n, num_channel)
            result_correlation_residual[:n, f] = np.asarray(
                correlation_residual[f].ndarray_aligned(axes_merit),
            ).reshape(n, num_channel)

        return self._result(
            images=images,
            solutions=result_solutions << unit,
            mean_chi_squared=result_chi2,
            correlation_residual=result_correlation_residual,
            num_iteration=num_iteration,
            success=np.array([r.success for r in results]),
            shape_scene=shape_scene,
            shape_channel=shape_channel,
            shape_frames=shape_frames,
        )

    def _result(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        solutions: u.Quantity,
        mean_chi_squared: np.ndarray,
        correlation_residual: np.ndarray,
        num_iteration: np.ndarray,
        success: np.ndarray,
        shape_scene: dict[str, int],
        shape_channel: dict[str, int],
        shape_frames: dict[str, int],
    ) -> IterativeInversionResult:
        """
        Gather the plain arrays of an inversion into its result.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction.
        solutions
            The solutions, an array with an axis for the iterations, the axes
            of the scene, and one axis for the frames.
        mean_chi_squared
            The mean chi squared, an array with an axis for the iterations,
            one for the frames and one for the channels.
            Only the iterations which any frame reached are kept.
        correlation_residual
            The correlation of the predicted images with the residuals, an
            array like `mean_chi_squared`.
        num_iteration
            The number of iterations of each frame.
        success
            Whether each frame converged.
        shape_scene
            The axes of the scene and their lengths.
        shape_channel
            The axes of the channels and their lengths.
        shape_frames
            The axes of the frames and their lengths.
        """
        axis_iteration = self.axis_iteration

        num = num_iteration.max()

        dims_scene = tuple(shape_scene.values())
        dims_channel = tuple(shape_channel.values())
        dims_frames = tuple(shape_frames.values())

        solutions = solutions.reshape((solutions.shape[0],) + dims_scene + dims_frames)

        shape_merit = (num,) + dims_frames + dims_channel
        mean_chi_squared = mean_chi_squared[:num].reshape(shape_merit)
        correlation_residual = correlation_residual[:num].reshape(shape_merit)

        axes_merit = (axis_iteration, *shape_frames, *shape_channel)

        if np.all(success):
            message = f"Achieved merit less than {self.threshold_convergence}."
        else:
            message = f"Max number of iterations ({self.num_iteration}) exceeded."

        if shape_frames:
            num_iteration = na.ScalarArray(
                ndarray=num_iteration.reshape(dims_frames),
                axes=tuple(shape_frames),
            )
            success = na.ScalarArray(
                ndarray=success.reshape(dims_frames),
                axes=tuple(shape_frames),
            )
        else:
            num_iteration = int(num_iteration[0])
            success = bool(success[0])

        return IterativeInversionResult(
            solutions=na.FunctionArray(
                inputs=self.instrument.coordinates_scene,
                outputs=na.ScalarArray(
                    ndarray=solutions,
                    axes=(axis_iteration, *shape_scene, *shape_frames),
                ),
            ),
            success=success,
            images=images,
            inverter=self,
            message=message,
            num_iteration=num_iteration,
            mean_chi_squared=na.ScalarArray(
                ndarray=mean_chi_squared << u.dimensionless_unscaled,
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
