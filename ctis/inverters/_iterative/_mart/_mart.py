import math
import warnings
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from ctis.instruments._operators import (
    Operators,
    aligned,
    framed,
    frames,
    unaligned,
)
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
    the instrument, which is slower, and a warning says so, with the reason.
    The two give the same result.
    A subclass of this inverter which replaces :meth:`mean_chi_squared` or
    :meth:`correlation_residual` runs on the methods of the instrument too,
    since the iteration on the matrices computes the merit itself.

    The images can have axes beyond those of the channels and the sensor,
    which are those of independent frames, sets of images of the same
    instrument.
    The scene of each frame is reconstructed on its own, and stops at its
    own iteration, exactly as it would alone, and the result has the axes of
    the frames, see :class:`~ctis.inverters.IterativeInversionResult`.
    A first guess is broadcast to the scene of every frame.
    One without the axes of the frames is the guess of every frame, and
    one without the axis of the wavelengths is the same at every wavelength.
    On the matrices, the frames are iterated on together and share each pass
    over them, which takes less time for each frame than a pass of its own.

    An axis which the instrument itself varies along, such as a time that
    its exposure varies along, is not an axis of frames, and neither is an
    axis named after one of the scene.
    The images are inverted along those in one iteration, which can only
    stop all at once, so they can have no more than one element along them.

    A first guess can be in any unit that the
    :meth:`~ctis.instruments.AbstractInstrument.image` method of the
    instrument takes a scene in, and the solution is given in the unit of
    the guess.
    It can be given with its coordinates, so the solution of one inversion
    can be the first guess of another.
    Every iteration multiplies the scene by a correction, so a voxel which
    is zero in the guess stays zero, and the solution of an inversion is
    zero in every voxel which a channel recorded no signal from.
    Such a solution cannot be the first guess of images which are bright
    where it is zero.
    Without a guess, the solution is in the unit that the
    :meth:`~ctis.instruments.AbstractInstrument.backproject` method of the
    instrument gives.
    """

    instrument: ctis.instruments.AbstractInstrument = dataclasses.MISSING
    """
    A model of a CTIS instrument which transforms the radiance of an observed
    scene to photons measured by the sensors.
    """

    gamma: float = 1
    r"""
    Learning rate, :math:`\gamma`.

    At every iteration, the correction of each channel, :math:`C`, is replaced
    by :math:`C^\gamma`, so the scene is multiplied by the geometric mean of
    the corrections of the channels, raised to the power :math:`\gamma`.

    With the default, 1, that is the geometric mean itself.
    A smaller learning rate takes proportionally more iterations to reach
    the same scene.

    A larger one takes fewer, but it can overshoot.
    Where one voxel alone lights the pixels it is imaged onto, its correction
    is the ratio of the scene to the current guess, so every iteration
    multiplies the error of its logarithm by :math:`1 - \gamma`.
    With a learning rate of 1 that voxel is corrected in one iteration,
    and with 2 it overshoots by as much as it was off, and never converges.
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

    Single precision is only available where the iteration runs on the
    sparse matrices of the instrument, and an inversion which cannot run on
    them raises an error, which says why it cannot, rather than running in
    double precision.
    """

    def __post_init__(self):

        # `None` is not a precision, though `numpy.dtype` takes it for one
        if self.dtype is None or np.dtype(self.dtype) not in (np.float32, np.float64):
            raise ValueError(
                f"`dtype` must be `numpy.float32` or `numpy.float64`, "
                f"got {self.dtype}."
            )

    def __call__(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: (
            None
            | na.ScalarArray
            | na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]
        ) = None,
        verbose: bool = False,
    ) -> IterativeInversionResult:
        """
        Reconstruct a scene using the observed images.

        Parameters
        ----------
        images
            The observed images used to calculate the reconstruction,
            as a function of their coordinates.
            Must be evaluated on the same position coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_sensor`
            attribute of :attr:`instrument`, to within the rounding of
            double precision.
            Any axes beyond those of the channels and the sensor are those of
            independent frames, sets of images of the same instrument.
            The scene of each frame is reconstructed on its own, and stops
            at its own iteration, exactly as it would alone.
            An axis which the instrument itself varies along, or one named
            after an axis of the scene, is not an axis of frames, and can
            have no more than one element.
        guess
            The initial guess at the reconstructed scene, the same for every
            frame, or one for each frame.
            It is broadcast to the scene of every frame: each of its axes
            should be an axis of the scene or of the frames, with as many
            elements as there, or with one, and it is the same along an
            axis which it does not have, or has one element along.
            So a single value is a guess, and so is a map without the axis
            of the wavelengths, which is the same at every wavelength.
            A guess with any other axis, or any other length, is refused.
            Must be evaluated on the same coordinates as
            :attr:`~ctis.instruments.AbstractInstrument.coordinates_scene`
            attribute of :attr:`instrument`.
            It can be given with its coordinates, as the
            :attr:`~ctis.inverters.IterativeInversionResult.solution` of
            an earlier inversion is, and they must then be those of the
            instrument, to within the rounding of double precision.
            A voxel which is zero in the guess is zero in the solution.
            It can be in any unit that the
            :meth:`~ctis.instruments.AbstractInstrument.image` method of
            :attr:`instrument` takes a scene in, and the solution is given
            in the unit of the guess.
        """

        instrument = self.instrument

        # the images are given with their coordinates, which are checked
        # here, once, and the iteration takes their values alone
        if not isinstance(images, na.AbstractFunctionArray):
            raise TypeError(
                f"The images should be given with their coordinates, as a "
                f"`FunctionArray`, but they are a `{type(images).__name__}`."
            )
        instrument._values_image(images, integrate=True, name="images")

        # a guess which is given with its coordinates, as the solution of an
        # earlier inversion is, has them checked here, once, and is inverted
        # as its values are
        if guess is not None:
            guess = instrument._values_scene(guess, name="guess")

        # the axes of the images beyond those of the channels and the sensor
        shape_extra = self._shape_extra(images.outputs)

        axis_iteration = self.axis_iteration
        if axis_iteration in shape_extra:
            raise ValueError(
                f"the images have an axis named {axis_iteration!r}, "
                f"which is the axis of the iterations in the result."
            )
        if axis_iteration in na.shape(guess):
            raise ValueError(
                f"the guess has an axis named {axis_iteration!r}, which is "
                f"the axis of the iterations in the result. The `solution` "
                f"of an earlier inversion does not have it, and its "
                f"`solutions` do."
            )

        arrays = self._arrays(images.outputs, guess)

        if not isinstance(arrays, str):
            return self._call_arrays(images, guess, verbose, *arrays)

        # the reason that this inversion cannot run on the matrices
        reason = arrays

        # the methods of an instrument compute in the precision of its own
        # arrays, whatever precision is asked for here
        dtype = np.dtype(self.dtype)
        if dtype != np.float64:
            raise ValueError(
                f"`dtype` is {dtype}, which is only available on the sparse "
                f"matrices of the instrument, and this inversion cannot run on "
                f"them: {reason}. It would run on the `image` and "
                f"`backproject` methods of the instrument, in double precision."
            )

        # the iteration on the methods of the instrument is much the slower
        # of the two, so it is never run without saying so, and why, before
        # the backprojection below, which is on those methods too
        warnings.warn(
            f"This inversion runs on the `image` and `backproject` methods of "
            f"the instrument, which is slower than on its sparse matrices: "
            f"{reason}.",
            stacklevel=2,
        )

        # the backprojection of one frame has every axis of the instrument
        # and of the scene, and none of the frames, and it is only needed
        # for images with axes of their own
        shape_frames = dict()
        shape_scene = None
        if shape_extra:
            frame = images.outputs[{ax: 0 for ax in shape_extra}]
            shape_scene = self._shape_scene(instrument.backproject(frame).outputs)
            shape_frames = {
                ax: num for ax, num in shape_extra.items() if ax not in shape_scene
            }

        # the other axes are those of the instrument or of the scene, which
        # the iteration below carries along, but cannot stop one at a time
        shape_own = {
            ax: num for ax, num in shape_extra.items() if ax not in shape_frames
        }
        if math.prod(shape_own.values()) > 1:
            raise ValueError(
                f"the images have the axes {shape_own}, which are also axes of "
                f"the instrument or of the scene, so they are not those of "
                f"independent frames, and the iteration on the methods of the "
                f"instrument, which cannot stop them one element at a time, can "
                f"take no more than one element along them. It runs on those "
                f"methods since it cannot run on the sparse matrices: {reason}."
            )

        if shape_frames:
            return self._call_frames(images, guess, verbose, shape_scene, shape_frames)

        return self._call_named(images, guess, verbose)

    def _shape_extra(self, images: na.AbstractScalar) -> dict[str, int]:
        """
        The axes of the images beyond those of the channels and the sensor,
        and their lengths.

        Parameters
        ----------
        images
            The values of the observed images.
        """
        instrument = self.instrument

        axis_images = (*instrument._axis_channel, *instrument.axis_sensor_xy)

        shape = na.shape(images)

        return {ax: shape[ax] for ax in shape if ax not in axis_images}

    def _shape_scene(self, backprojected: na.AbstractScalar) -> dict[str, int]:
        """
        The axes of the scene of one frame and their lengths, for an
        inversion which runs on the methods of the instrument, with any axes
        which the instrument itself has, such as a time that its exposure
        varies along.

        They are the axes of the backprojection of the frame, which has
        every axis of the instrument and of the scene, and none of the
        frames, without those of the channels.
        The axes of the frames are therefore those of the images, beyond
        those of the channels and the sensor, which are not among these.

        Parameters
        ----------
        backprojected
            The values of the backprojection of the images of one frame.
        """
        axis_channel = self.instrument._axis_channel

        shape = na.shape(backprojected)

        return {ax: num for ax, num in shape.items() if ax not in axis_channel}

    @staticmethod
    def _check_guess(
        guess: na.AbstractScalar,
        shape_scene: dict[str, int],
        shape_frames: dict[str, int],
    ) -> None:
        """
        Check that a guess can be broadcast to the scene of every frame:
        that each of its axes is an axis of the scene or of the frames, with
        one element or with as many as there.

        Both paths of the inverter broadcast such a guess, and neither can
        invert any other, so both check it here, before they image anything.

        Parameters
        ----------
        guess
            The values of the initial guess at the reconstructed scene.
        shape_scene
            The axes of the scene and their lengths.
        shape_frames
            The axes of the frames and their lengths.
        """
        shape = na.shape(guess)

        shape_all = shape_scene | shape_frames

        valid = all(
            ax in shape_all and num in (1, shape_all[ax]) for ax, num in shape.items()
        )

        if valid:
            return

        frames = f" or of the frames, {shape_frames}" if shape_frames else ""

        raise ValueError(
            f"Every axis of the guess should be an axis of the scene, "
            f"{shape_scene}{frames}, with one element or with as many, but "
            f"its axes are {shape}."
        )

    def _arrays(
        self,
        images: na.AbstractScalar,
        guess: None | na.AbstractScalar,
    ) -> (
        str
        | tuple[
            dict[str, int],
            Operators,
            np.ndarray,
            None | np.ndarray,
            None | np.ndarray,
        ]
    ):
        """
        The axes of the frames, and the operators of the instrument with the
        observed images and the initial guess as the plain arrays that the
        operators act on, or the reason that any of them cannot be
        expressed that way, which is said in the warning that the
        inversion then runs on the methods of the instrument.

        Each array has the frames along one last axis.
        The guess is a copy, which the iteration is free to overwrite, or
        :obj:`None` if there is none.

        A guess in units of energy, for operators which take a scene in
        photons, is divided by the energy of the photons of each voxel, if
        the operators have it, which is how the instrument itself takes
        such a scene.
        That energy is the last of the results, for the solution to be
        multiplied by, and it is :obj:`None` for any other guess.

        A guess which cannot be broadcast to the scene of every frame raises
        an error, see :meth:`_check_guess`, since the methods of the
        instrument cannot invert it either.

        Parameters
        ----------
        images
            The values of the observed images.
        guess
            The initial guess at the reconstructed scene, if any.
        """
        # an inverter which replaces a merit function is left to it, since
        # the iteration on the operators computes the merit itself
        cls = type(self)
        if (
            cls.mean_chi_squared is not MartInverter.mean_chi_squared
            or cls.correlation_residual is not MartInverter.correlation_residual
        ):
            return (
                f"the inverter, a `{cls.__name__}`, replaces a merit function, "
                f"which the iteration on the matrices computes itself"
            )

        instrument = self.instrument

        operators = instrument._operators(self.dtype)
        if operators is None:
            reason = instrument._operators_reason(self.dtype)
            return f"the instrument provides none, since {reason}"

        # every axis of the images beyond their own is an axis of the frames,
        # unless it is one of those of the scene
        shape_frames = frames(
            shape=na.shape(images),
            shape_own=operators.shape_images,
            shape_other=operators.shape_scene,
        )
        if shape_frames is None:
            axes = tuple(
                ax
                for ax in na.shape(images)
                if ax in operators.shape_scene and ax not in operators.shape_images
            )
            return (
                f"the images have the axes {axes}, which are axes of the scene, "
                f"so they are not a stack of frames"
            )

        # a guess which cannot be broadcast to the scene of every frame is
        # refused here, since the methods of the instrument cannot invert
        # it either, and would backproject a frame before they refused it
        if guess is not None:
            self._check_guess(guess, operators.shape_scene, shape_frames)

        num_frame = math.prod(shape_frames.values())

        shape_scene = tuple(operators.shape_scene.values())

        observed = framed(
            a=images,
            shape=operators.shape_images,
            shape_frames=shape_frames,
            unit=operators.unit_images,
            dtype=operators.dtype,
        )
        if observed is None:
            reason = unaligned(
                a=images,
                shape=operators.shape_images | shape_frames,
                unit=operators.unit_images,
            )
            return f"the array of the images {reason}"

        scene = None
        energy = None
        if guess is not None:
            shape_guess = operators.shape_scene | shape_frames
            unit_energy = operators.unit_scene * operators.unit_energy

            scene = aligned(
                a=guess,
                shape=shape_guess,
                unit=operators.unit_scene,
                dtype=operators.dtype,
            )

            # a guess in units of energy, for operators which take a scene
            # in photons, is divided by the energy of the photons of each
            # voxel below
            if scene is None and operators.energy is not None:
                energy = operators.energy
                scene = aligned(
                    a=guess,
                    shape=shape_guess,
                    unit=unit_energy,
                    dtype=operators.dtype,
                )

            # a guess which is not a plain array of the scene, in a unit
            # that the operators take one in, is left to the named arrays
            if scene is None:
                if unaligned(guess, shape=shape_guess, unit=unit_energy) is None:
                    return (
                        "the guess is in units of energy, which the matrices of "
                        "the instrument do not take a scene in"
                    )
                reason = unaligned(
                    a=guess,
                    shape=shape_guess,
                    unit=operators.unit_scene,
                )
                return f"the guess {reason}"

            # the guess is the same along an axis of the scene or of the
            # frames which it does not vary along
            scene = np.broadcast_to(scene, shape_scene + tuple(shape_frames.values()))
            scene = np.array(scene, dtype=operators.dtype, order="C")
            scene = scene.reshape(shape_scene + (num_frame,))

            if energy is not None:
                scene /= energy

        return shape_frames, operators, observed, scene, energy

    def _call_arrays(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.ScalarArray,
        verbose: bool,
        shape_frames: dict[str, int],
        operators: Operators,
        observed: np.ndarray,
        scene: None | np.ndarray,
        energy: None | np.ndarray,
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
        energy
            The energy of the photons of each voxel, if `guess` is in units
            of energy and was divided by it to give `scene`, which the
            solution is then multiplied by, or :obj:`None` otherwise.
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

        # the merit of each frame and each channel, for every iteration
        # which has run
        chi2 = []
        correlation_residual = []

        intermediate = []

        # the photons leaving every voxel of the frames which are still
        # iterating, which every iteration overwrites
        buffer = None

        for i in range(self.num_iteration):

            if self.intermediate:
                # a frame which has stopped keeps its solution
                solution[:, active] = scene
                intermediate.append(solution.copy())

            if verbose:  # pragma: nocover
                print(f"{i=}")

            if buffer is None or buffer.shape[~0] != active.size:
                shape_buffer = operators.forward.shape_input + (active.size,)
                buffer = np.empty(shape_buffer, dtype=dtype)

            predicted, variance = operators.image(
                scene=scene.reshape(shape_scene + (active.size,)),
                buffer=buffer,
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

            # a frame has no merit once it has stopped
            chi2.append(np.full((num_frame, num_channel), np.nan))
            chi2[~0][active] = chi2_i
            correlation_residual.append(np.full((num_frame, num_channel), np.nan))
            correlation_residual[~0][active] = correlation_residual_i

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
        if energy is not None:
            # the solution of a guess in units of energy, which was divided
            # by the energy of the photons of each voxel, is multiplied by
            # it again, in one pass with the factor between the two units
            # of energy
            unit = operators.unit_scene * operators.unit_energy
            solutions *= energy * unit.to(guess.unit)
            solutions = solutions << guess.unit
        else:
            solutions = solutions << operators.unit_scene
            if guess is not None:
                solutions = solutions.to(guess.unit, copy=False)

        shape_merit = (len(chi2), num_frame, num_channel)

        return self._result(
            images=images,
            solutions=solutions,
            mean_chi_squared=np.array(chi2).reshape(shape_merit),
            correlation_residual=np.array(correlation_residual).reshape(shape_merit),
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
        shape_scene: dict[str, int],
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
        shape_scene
            The axes of the scene of one frame and their lengths, see
            :meth:`_shape_scene`.
        shape_frames
            The axes of the frames and their lengths.
        """
        axis_iteration = self.axis_iteration

        if guess is not None:
            # the guess is checked before any frame is inverted, and it is
            # the guess of every frame along an axis of the frames which it
            # does not have, or has one element along
            self._check_guess(guess, shape_scene, shape_frames)
            guess = na.broadcast_to(guess, na.shape(guess) | shape_frames)

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

        backprojected = instrument.backproject(images).outputs

        if guess is None:
            scene = backprojected.mean(axis_channel)
            scene.ndarray[:] = scene.ndarray.mean()
        else:
            # the guess is broadcast to the scene, which the backprojection
            # has every axis of, into an array of its own, which the
            # iteration is free to overwrite
            shape_scene = self._shape_scene(backprojected)
            self._check_guess(guess, shape_scene, dict())
            scene = na.broadcast_to(guess, na.shape(guess) | shape_scene).copy()

        num_channel = instrument.num_channel

        gamma = self.gamma

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
