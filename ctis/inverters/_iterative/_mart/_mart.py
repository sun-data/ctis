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

    unit: None | u.UnitBase = None
    """
    The unit of the reconstructed scene.

    This is forwarded to
    :meth:`~ctis.instruments.AbstractInstrument.backproject`, which expresses
    the result in either photon or energy units as requested.
    If :obj:`None` (the default), the natural units of the backprojection are
    used, which differ between instruments.

    The iteration itself is unaffected, since the multiplicative correction is
    a ratio of two backprojections and is therefore dimensionless.
    """

    regularization: float = 0
    r"""
    The weight, :math:`\beta`, of a smoothness penalty applied to the
    reconstructed scene along :attr:`axis_regularization`.

    After every multiplicative correction, the current guess takes one
    gradient-descent step on the penalty

    .. math::

        R(\hat{u}) = \frac{1}{2} \sum_k \left( \hat{u}_{k+1} - \hat{u}_k \right)^2,

    where :math:`k` indexes the cells along a regularized axis,

    .. math::

        \hat{u} \leftarrow \hat{u} - \beta \nabla R(\hat{u}),

    which is the same as convolving :math:`\hat{u}` with the kernel
    :math:`[\beta, 1 - 2 \beta, \beta]` along that axis.
    If there is more than one regularized axis, the step is taken along each
    axis in turn, which is a convolution with the separable kernel.
    This is the smoothed-EM strategy of :cite:t:`Silverman1990` applied to
    MART, and :math:`\beta = 1/4` reproduces the smoothing kernel of
    :cite:t:`Parker2022` exactly when all three axes of the scene are
    regularized.

    The step conserves the sum of :math:`\hat{u}` along each regularized axis,
    so the radiance integrated over wavelength is unchanged in every spatial
    pixel, and it preserves positivity as long as
    :math:`0 \leq \beta \leq 1 / 2`.
    Values outside this range raise a :class:`ValueError`.

    If zero (the default), the reconstruction is unregularized.
    """

    axis_regularization: None | str | tuple[str, ...] = None
    """
    The logical axes of the scene along which :attr:`regularization`
    is applied.

    If :obj:`None` (the default), only the wavelength axis of the instrument,
    :attr:`~ctis.instruments.AbstractInstrument.axis_wavelength`,
    is regularized, since a CTIS with only a few channels constrains the
    spectral direction of the scene much more weakly than the spatial
    directions.
    Any of the wavelength axis and the two spatial axes of the scene,
    :attr:`~ctis.instruments.AbstractInstrument.axis_scene_xy`,
    may be given.
    """

    def __post_init__(self):

        if self.gamma is None:
            self.gamma = 2 / self.instrument.num_channel

        instrument = self.instrument
        axis_regularization = self.axis_regularization_
        axis_valid = (instrument.axis_wavelength, *instrument.axis_scene_xy)
        for axis in axis_regularization:
            if axis not in axis_valid:
                raise ValueError(
                    f"`axis_regularization` must be a subset of {axis_valid}, "
                    f"got {axis_regularization!r}."
                )

        if not (0 <= self.regularization <= 1 / 2):
            raise ValueError(
                f"`regularization` must be between 0 and 1/2 to preserve "
                f"positivity, got {self.regularization!r}."
            )

    @property
    def axis_regularization_(self) -> tuple[str, ...]:
        """
        :attr:`axis_regularization` normalized to a tuple of axis names,
        with :obj:`None` resolved to the wavelength axis of the instrument.
        """
        axis = self.axis_regularization
        if axis is None:
            axis = self.instrument.axis_wavelength
        if isinstance(axis, str):
            axis = (axis,)
        return tuple(axis)

    def regularize(self, scene: na.ScalarArray) -> na.ScalarArray:
        r"""
        Take one gradient-descent step of size :attr:`regularization`
        on the smoothness penalty along :attr:`axis_regularization`.

        Parameters
        ----------
        scene
            The current guess at the reconstructed scene.
        """
        beta = self.regularization

        if beta == 0:
            return scene

        result = scene.copy()

        for axis in self.axis_regularization_:
            # the flux moved across each cell interface, from the higher
            # cell into the lower cell.
            # Since every interface moves flux from one cell into another,
            # the sum along `axis` is conserved exactly.
            flux = beta * np.diff(result, axis=axis)
            result[{axis: slice(None, ~0)}] += flux
            result[{axis: slice(1, None)}] -= flux

        return result

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
            scene = instrument.backproject(images, unit=self.unit).outputs
            scene = scene.mean(axis_channel)
            scene.ndarray[:] = scene.ndarray.mean()
        else:
            scene = guess.copy()

        num_channel = instrument.num_channel

        gamma = self.gamma

        backprojected = instrument.backproject(images, unit=self.unit).outputs

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

            backprojected_new = instrument.backproject(
                images_new,
                unit=self.unit,
            ).outputs

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

            scene = self.regularize(scene)

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
