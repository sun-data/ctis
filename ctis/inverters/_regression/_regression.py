from typing import TYPE_CHECKING
import abc
import warnings
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from ..._torch import _torch
from .._iterative import AbstractIterativeInverter, IterativeInversionResult

__all__ = [
    "AbstractRegressionInverter",
    "ElasticNetInverter",
    "RidgeInverter",
    "LassoInverter",
    "RegressionInversionResult",
]

if TYPE_CHECKING:  # pragma: nocover
    import torch


def _sum(a: "torch.Tensor") -> float:
    """
    Sum every element of a tensor in double precision.

    The merit function and the duality gap are differences of large sums,
    which lose most of their significant figures if accumulated in single
    precision.
    """
    return a.double().sum().item()


def _pearsonr(
    a: "torch.Tensor",
    b: "torch.Tensor",
    dim: tuple[int, ...],
) -> "torch.Tensor":
    """Pearson's correlation coefficient of two tensors along `dim`."""
    a = a - a.mean(dim, keepdim=True)
    b = b - b.mean(dim, keepdim=True)
    norm = ((a * a).sum(dim) * (b * b).sum(dim)).sqrt()
    return (a * b).sum(dim) / norm


@dataclasses.dataclass
class AbstractRegressionInverter(
    AbstractIterativeInverter,
):
    """
    An abstract inversion algorithm which reconstructs an observed scene
    using penalized linear regression.

    Since the forward model of a
    :class:`~ctis.instruments.AbstractLinearInstrument` is a matrix
    multiplication, reconstructing the scene is a linear regression in which
    every voxel of the scene is a coefficient and every pixel of the sensors is
    a sample.
    There are many more voxels than pixels, so the regression is
    underdetermined, and these algorithms choose among the scenes which are
    consistent with the measurement by penalizing the radiance of the scene.

    Unlike :class:`~ctis.inverters.MartInverter`, where the initial guess
    determines which of the consistent scenes is found, the answer found by
    these algorithms does not depend on where the iteration starts.
    Any prior information must be supplied explicitly.
    """

    @property
    @abc.abstractmethod
    def alpha(self) -> float:
        """The overall strength of the penalty."""

    @property
    @abc.abstractmethod
    def l1_ratio(self) -> float:
        r"""
        The fraction of the penalty which is :math:`\ell_1`,
        the remainder being :math:`\ell_2`.
        """


@dataclasses.dataclass
class ElasticNetInverter(
    AbstractRegressionInverter,
):
    r"""
    Reconstruct a scene using nonnegative elastic-net regression
    :cite:p:`Zou2005`, solved using the fast iterative shrinkage-thresholding
    algorithm (FISTA) :cite:p:`Beck2009`.

    The reconstruction, :math:`\hat{x}`, minimizes the merit function

    .. math::

        F(x) = \frac{1}{2} \left\langle
            \left( \frac{d - A x}{\sigma} \right)^2
        \right\rangle
        + \alpha \left[
            \rho \left\langle c \, \frac{x}{s} \right\rangle
            + \frac{1 - \rho}{2} \left\langle
                \left( \frac{x - x_0}{s} \right)^2
            \right\rangle
        \right],
        \quad x \geq 0,

    where :math:`d` are the measured electrons,
    :math:`A` is the forward model of :attr:`instrument`,
    :math:`\sigma` is the standard deviation of the measurement noise,
    :math:`\alpha` is :attr:`alpha`,
    :math:`\rho` is :attr:`l1_ratio`,
    :math:`c` are the :attr:`penalty_weights`,
    and :math:`x_0` is the :attr:`prior`.
    The first average is over the pixels of the sensors and is half of the
    mean chi squared.
    The other two averages are over the voxels of the scene.
    The reference radiance, :math:`s`, is the uniform scene which reproduces
    the total measured signal.
    It makes the penalty dimensionless, so that the same :math:`\alpha`
    means the same thing for any instrument, exposure time, or grid.

    Every term of the merit function is convex,
    and if :math:`\rho < 1` it has exactly one minimum,
    which is found regardless of the initial guess.

    The forward model of :attr:`instrument` is assembled into a sparse
    :class:`~ctis.Regridder`, so the whole iteration can run on a GPU.
    Each iteration applies the forward model once and its exact transpose once.
    :meth:`~ctis.instruments.AbstractInstrument.backproject` is never used.

    The step taken by each voxel is set by a separable quadratic surrogate of
    the first term of the merit function :cite:p:`Erdogan1999`,
    so there is no step size to tune, and the merit function never increases.
    The number of iterations needed grows as :math:`\alpha` shrinks and as
    the signal-to-noise ratio of the measurement grows.

    Notes
    -----
    Since the radiance is nonnegative, the :math:`\ell_1` norm is just a sum,
    and the :math:`\ell_1` penalty is equivalent to shifting the prior of the
    :math:`\ell_2` penalty downward by :math:`\rho c s / (1 - \rho)`.
    It is the constraint :math:`x \geq 0` which then sets voxels to exactly
    zero.

    If the solution is expressed in units of :math:`s`, there is no prior,
    and the penalty weights are unity, this merit function is the same as the
    one minimized by :class:`sklearn.linear_model.ElasticNet` with
    ``positive=True``, ``sample_weight`` :math:`\propto 1 / \sigma^2`,
    ``l1_ratio`` :math:`= \rho`, and ``alpha`` :math:`= \alpha / N`,
    where :math:`N` is the number of voxels in the scene.

    Examples
    --------

    Reconstruct a scene of randomly-placed Gaussians observed by an idealized
    CTIS instrument.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import ctis

        # Define the grid of velocities and positions on the skyplane.
        wavelength_rest = 630 * u.AA
        velocity = na.linspace(-250, 250, axis="wavelength", num=11) * u.km / u.s
        coordinates_scene = na.DopplerPositionalVectorArray.from_velocity(
            velocity=velocity,
            wavelength_rest=wavelength_rest,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
                num=33,
            ),
        )

        # Define the grid of positions on the sensor.
        coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
            velocity=velocity,
            wavelength_rest=wavelength_rest,
            position=na.Cartesian2dVectorArray(
                x=na.arange(0, 65, axis="sensor_x") * u.pix,
                y=na.arange(0, 65, axis="sensor_y") * u.pix,
            ),
        )

        # Define an idealized CTIS instrument with four channels.
        angle = na.linspace(0, 360, num=4, axis="channel", endpoint=False) * u.deg
        instrument = ctis.instruments.IdealInstrument(
            area_effective=1 * u.cm**2,
            timedelta_exposure=20 * u.s,
            plate_scale=0.625 * u.arcsec / u.pix,
            dispersion=0.105 * u.AA / u.pix,
            angle=angle,
            wavelength_ref=wavelength_rest,
            position_ref=32 * u.pix,
            coordinates_scene=coordinates_scene,
            coordinates_sensor=coordinates_sensor,
            channel=angle,
            axis_channel="channel",
            axis_wavelength="wavelength",
            axis_scene_xy=("scene_x", "scene_y"),
            axis_sensor_xy=("sensor_x", "sensor_y"),
        )

        # Observe a scene of randomly-placed Gaussians.
        scene = ctis.scenes.gaussians(coordinates_scene)
        images = instrument.image(scene)

        # Reconstruct the scene.
        inverter = ctis.inverters.ElasticNetInverter(
            instrument=instrument,
            alpha=1e-2,
            l1_ratio=0.5,
            num_iteration=2000,
        )
        result = inverter(images)

        # Compare the true and reconstructed radiance,
        # integrated over wavelength.
        axis = "wavelength"
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=2,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            na.plt.pcolormesh(
                scene.inputs.position,
                C=na.value(scene.outputs.sum(axis)),
                ax=ax[0],
            )
            na.plt.pcolormesh(
                result.solution.inputs.position,
                C=na.value(result.solution.outputs.sum(axis)),
                ax=ax[1],
            )
            ax[0].set_title("truth")
            ax[1].set_title("reconstruction")
    """

    instrument: ctis.instruments.AbstractLinearInstrument = dataclasses.MISSING
    """
    A model of a CTIS instrument which transforms the radiance of an observed
    scene to electrons measured by the sensors.
    """

    alpha: float = 1e-2
    r"""
    The overall strength of the penalty, :math:`\alpha`.

    This is dimensionless, and is the price of the penalty in units of the
    mean chi squared: the reconstruction is allowed to fit the measurement
    worse by about this much in exchange for a smaller penalty.
    If zero, the reconstruction is a nonnegative least-squares fit.
    """

    l1_ratio: float = 0.5
    r"""
    The fraction of the penalty which is :math:`\ell_1`, :math:`\rho`.

    If zero the penalty is purely :math:`\ell_2` (ridge regression),
    which spreads the radiance over all of the voxels consistent with the
    measurement, and if one it is purely :math:`\ell_1` (lasso regression),
    which concentrates the radiance into as few voxels as possible.
    """

    prior: None | u.Quantity | na.AbstractScalar | na.AbstractFunctionArray = None
    r"""
    The scene toward which the :math:`\ell_2` penalty pulls the
    reconstruction, :math:`x_0`.

    This must be broadcastable to the voxels of
    :attr:`~ctis.instruments.AbstractInstrument.coordinates_scene`
    and convertible to :attr:`unit_scene`.

    The prior only chooses among the scenes which fit the measurement
    comparably well.
    As :math:`\alpha` goes to zero, the reconstruction approaches the scene
    closest to the prior out of those which best fit the measurement.

    If :obj:`None` (the default), the prior is zero.
    """

    penalty_weights: None | na.AbstractScalar = None
    r"""
    The relative cost of radiance in each voxel under the :math:`\ell_1`
    penalty, :math:`c`.

    This must be nonnegative, dimensionless, and broadcastable to the voxels of
    :attr:`~ctis.instruments.AbstractInstrument.coordinates_scene`.
    A voxel with a larger weight is more expensive to fill,
    which can be used to suggest `where` the radiance is without suggesting
    `how much` of it there is, for example using the reciprocal of a context
    image :cite:p:`Zou2006`.

    If :obj:`None` (the default), every voxel has unit weight.
    """

    threshold_convergence: float = 1e-4
    r"""
    The convergence threshold, :math:`T`, which halts the iteration.

    The iteration is converged once the duality gap,
    an upper bound on the difference between the current merit function
    and its minimum, is smaller than :math:`T` times the merit function.

    If :math:`\rho = 1` the duality gap is only a tight bound very close to
    the minimum, so the iteration is also considered converged once the merit
    function decreases by less than :math:`T` times its value between
    evaluations of the duality gap.
    """

    unit: None | u.UnitBase = None
    """
    The unit of the reconstructed scene.

    If :obj:`None` (the default), :attr:`unit_scene` is used.
    """

    num_iteration: int = dataclasses.field(default=1000, kw_only=True)
    """
    The maximum number of iterations to perform.

    If convergence is not reached before this number is exceeded,
    a warning is raised and an unsuccessful result is returned.
    """

    num_iteration_check: int = dataclasses.field(default=10, kw_only=True)
    """
    The number of iterations between evaluations of the duality gap.

    Each evaluation costs an extra application of the transpose of the forward
    model.
    """

    uncertainty: None | na.AbstractScalar = dataclasses.field(
        default=None,
        kw_only=True,
    )
    r"""
    The standard deviation of the measurement noise, :math:`\sigma`,
    in electrons.

    If :obj:`None` (the default) and `images` carries an uncertainty, as
    produced by ``instrument.image(uncertainty=True)``, that uncertainty is
    used.
    Otherwise the instrument's own noise model is evaluated on the `measured`
    signal: the variance is taken to be the read noise plus a term
    proportional to the measurement, with both coefficients found by asking
    the instrument for the uncertainty of an empty and of a uniform scene.
    A pixel which measured less than one photon is assigned the variance of
    one photon, otherwise empty pixels would carry no weight at all.

    The uncertainty is held fixed during the iteration.
    To use the uncertainty predicted by a previous reconstruction instead,
    which is less biased for faint signals, evaluate
    ``instrument.image(result.solution, noise=False, uncertainty=True)``
    and provide its width here.
    """

    variance_min: float = dataclasses.field(default=0, kw_only=True)
    """
    A lower bound on the variance, in electrons squared.

    A pixel whose variance is zero is ignored.
    """

    device: None | str = dataclasses.field(default=None, kw_only=True)
    """
    The :mod:`torch` device on which to perform the iteration.

    If :obj:`None`, a CUDA device is used if one is available.
    """

    dtype: "None | torch.dtype" = dataclasses.field(default=None, kw_only=True)
    """
    The floating-point type used during the iteration.

    If :obj:`None`, :obj:`torch.float32` is used.
    """

    def __post_init__(self):

        if self.alpha < 0:
            raise ValueError(f"`alpha` must be nonnegative, got {self.alpha!r}.")

        if not (0 <= self.l1_ratio <= 1):
            raise ValueError(
                f"`l1_ratio` must be between 0 and 1, got {self.l1_ratio!r}."
            )

        instrument = self.instrument
        if not isinstance(instrument, ctis.instruments.AbstractLinearInstrument):
            raise ValueError(
                f"{type(instrument)=} is not supported, the forward model must "
                f"be a `ctis.instruments.AbstractLinearInstrument`."
            )

    @property
    def unit_scene(self) -> u.UnitBase:
        """
        The natural unit of the reconstructed scene, a spectral radiance.

        This is derived from the units of
        :attr:`~ctis.instruments.AbstractLinearInstrument.response`, so an
        instrument whose forward model consumes a photon radiance yields a
        photon radiance, and one which consumes an energy radiance yields an
        energy radiance.
        """
        scale_input, scale_output = self.instrument.response
        unit = na.unit_normalized(scale_input) * na.unit_normalized(scale_output)
        result = u.electron / unit

        # express the result in a conventional radiance unit if possible
        for candidate in (
            u.erg / (u.cm**2 * u.sr * u.s * u.AA),
            u.ph / (u.cm**2 * u.sr * u.s * u.AA),
        ):
            if result.is_equivalent(candidate):
                return candidate

        return result  # pragma: nocover

    @property
    def _unit(self) -> u.UnitBase:
        """The unit of the reconstructed scene, with the default resolved."""
        if self.unit is not None:
            return self.unit
        return self.unit_scene

    @property
    def _axes_scene(self) -> tuple[str, ...]:
        """The logical axes of the tensors which represent a scene."""
        instrument = self.instrument
        return (instrument.axis_wavelength, *instrument.axis_scene_xy)

    def _regridder(self) -> "ctis.Regridder":
        """Assemble the weights of the instrument into a sparse operator."""
        instrument = self.instrument
        return ctis.Regridder.from_weights(
            weights=instrument.weights,
            axis_input=tuple(instrument.axis_scene_xy),
            axis_output=tuple(instrument.axis_sensor_xy),
            device=self.device,
            dtype=self.dtype,
        )

    def _axes_images(self, regridder: "ctis.Regridder") -> tuple[str, ...]:
        """The logical axes of the tensors which represent the images."""
        instrument = self.instrument
        axis_wavelength = instrument.axis_wavelength
        axes = tuple(ax for ax in regridder.axis_block if ax != axis_wavelength)
        return axes + tuple(instrument.axis_sensor_xy)

    def _operator(self, regridder: "ctis.Regridder"):
        r"""
        The noiseless forward model of the instrument and its exact transpose,
        as a pair of functions which operate on :mod:`torch` tensors.

        The first function maps a scene in units of :attr:`unit`,
        with axes :attr:`_axes_scene`, into images in electrons,
        with axes :meth:`_axes_images`.
        The second function is the transpose of the first.
        """
        torch = _torch()

        instrument = self.instrument
        axis_wavelength = instrument.axis_wavelength

        device = regridder.device
        dtype = regridder.dtype
        axis_block = regridder.axis_block

        index_wavelength = axis_block.index(axis_wavelength)
        index_other = tuple(i for i in range(len(axis_block)) if i != index_wavelength)

        scale_input, scale_output = instrument.response

        # attach the unit of the scene to the first factor and express it in
        # whichever unit makes the second factor yield electrons.
        scale_input = scale_input * self._unit
        scale_input = scale_input.to(u.electron / na.unit_normalized(scale_output))

        # align both factors to the layout of the regridder, leaving axes of
        # unit length to be broadcast, since the second factor is otherwise
        # as large as the unintegrated images.
        scale_input = na.value(
            na.as_named_array(scale_input).ndarray_aligned(regridder.axes_values_input)
        )
        scale_output = na.value(
            na.as_named_array(scale_output).ndarray_aligned(
                regridder.axes_values_output
            )
        )

        def _tensor(array: np.ndarray) -> "torch.Tensor":
            return torch.as_tensor(np.ascontiguousarray(array)).to(
                device=device,
                dtype=dtype,
            )

        scale_input = _tensor(scale_input)
        scale_output = _tensor(scale_output)

        shape_input = regridder.shape_values_input
        shape_output = regridder.shape_values_output

        # the scene does not depend on the block axes other than wavelength
        shape_view = tuple(
            shape_input[i] if i == index_wavelength else 1
            for i in range(len(axis_block))
        )
        shape_view = shape_view + shape_input[len(axis_block) :]

        def forward(scene: "torch.Tensor") -> "torch.Tensor":
            cube = scene.reshape(shape_view) * scale_input
            cube = cube.expand(shape_input)
            result = regridder(cube) * scale_output
            return result.sum(index_wavelength)

        def transpose(images: "torch.Tensor") -> "torch.Tensor":
            cube = images.unsqueeze(index_wavelength) * scale_output
            cube = cube.expand(shape_output)
            result = regridder.adjoint(cube) * scale_input
            if index_other:
                result = result.sum(index_other)
            return result

        return forward, transpose

    def _array_scene(
        self,
        regridder: "ctis.Regridder",
        a: u.Quantity | na.AbstractScalar | na.AbstractFunctionArray,
        unit: None | u.UnitBase,
    ) -> np.ndarray:
        """
        Broadcast a quantity defined on the voxels of the scene into a plain
        array with axes :attr:`_axes_scene`.
        """
        if isinstance(a, na.AbstractFunctionArray):
            a = a.outputs
        axes = self._axes_scene
        shape = {ax: regridder.shape_input[ax] for ax in axes}
        a = na.broadcast_to(na.as_named_array(a), shape)
        a = u.Quantity(a.ndarray_aligned(axes))
        if unit is None:
            return a.to_value(u.dimensionless_unscaled)
        return a.to_value(unit)

    def _scene(self, outputs: np.ndarray) -> na.FunctionArray:
        """Express a plain array with axes :attr:`_axes_scene` as a scene."""
        return na.FunctionArray(
            inputs=self.instrument.coordinates_scene,
            outputs=na.ScalarArray(
                ndarray=outputs << self._unit,
                axes=self._axes_scene,
            ),
        )

    def _variance(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        data: np.ndarray,
        axes: tuple[str, ...],
        scene_uniform: np.ndarray,
    ) -> np.ndarray:
        """
        The variance of every measurement, in electrons squared.

        Parameters
        ----------
        images
            The observed images.
        data
            The nominal value of the observed images in electrons,
            aligned to `axes`.
        axes
            The logical axes of the result.
        scene_uniform
            A uniform scene which reproduces the total measured signal.
        """
        outputs = images.outputs

        if self.uncertainty is not None:
            width = na.as_named_array(self.uncertainty).ndarray_aligned(axes)
            variance = np.square(u.Quantity(width).to_value(u.electron))
        elif isinstance(outputs, na.AbstractUncertainScalarArray):
            width = outputs.width.ndarray_aligned(axes)
            variance = np.square(u.Quantity(width).to_value(u.electron))
        else:
            instrument = self.instrument

            def _noise(scene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                result = instrument.image(
                    self._scene(scene),
                    noise=False,
                    uncertainty=True,
                ).outputs
                nominal = u.Quantity(result.nominal.ndarray_aligned(axes))
                width = u.Quantity(result.width.ndarray_aligned(axes))
                return nominal.to_value(u.electron), width.to_value(u.electron)

            # the variance of shot noise is proportional to the signal, so
            # two evaluations of the noise model determine it for any signal.
            _, width_empty = _noise(np.zeros_like(scene_uniform))
            nominal, width = _noise(scene_uniform)

            variance_read = np.square(width_empty)
            where = nominal > 0
            slope = np.divide(
                np.square(width) - variance_read,
                nominal,
                out=np.zeros_like(nominal),
                where=where,
            )
            slope = np.where(where, slope, np.max(slope, initial=0))

            # the slope is also the signal of a single photon, which is the
            # smallest signal the shot noise can be estimated from.
            variance = variance_read + slope * np.maximum(data, slope)

        return np.maximum(variance, self.variance_min)

    def __call__(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.AbstractScalar | na.AbstractFunctionArray = None,
        verbose: bool = False,
    ) -> "RegressionInversionResult":
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
            The scene from which to start the iteration.
            This only affects how many iterations are needed, not the result,
            so it is useful for continuing from a previous reconstruction.
            If :obj:`None` (the default), the iteration starts from
            :attr:`prior`, or from a uniform scene if there is no prior.
        verbose
            Whether to print the merit function at every iteration.
        """
        torch = _torch()

        instrument = self.instrument

        position_images = images.inputs.position
        position_sensor = instrument.coordinates_sensor.position
        if not np.all(position_images == position_sensor):
            raise ValueError(
                "`images.inputs.position` and `self.coordinates_sensor.position` "
                "are not equal."
            )

        regridder = self._regridder()
        device = regridder.device
        dtype = regridder.dtype

        def _tensor(array: np.ndarray) -> "torch.Tensor":
            return torch.as_tensor(np.ascontiguousarray(array)).to(
                device=device,
                dtype=dtype,
            )

        axes_scene = self._axes_scene
        axes_images = self._axes_images(regridder)

        shape_scene = tuple(regridder.shape_input[ax] for ax in axes_scene)
        dim_sensor = tuple(range(len(axes_images) - 2, len(axes_images)))

        operator, operator_transpose = self._operator(regridder)

        outputs = images.outputs
        for ax in outputs.shape:
            if ax not in axes_images:
                raise ValueError(
                    f"`images` has an axis, {ax!r}, which is not one of the "
                    f"axes of the instrument, {axes_images}."
                )
        data = na.nominal(outputs)
        data = u.Quantity(data.ndarray_aligned(axes_images)).to_value(u.electron)
        data = np.broadcast_to(
            data, tuple(regridder.shape_output[ax] for ax in axes_images)
        )

        # the uniform scene which reproduces the total measured signal sets
        # the scale of the problem, so the penalty is dimensionless and the
        # iteration works with numbers of order unity.
        ones = torch.ones(shape_scene, device=device, dtype=dtype)
        total_ones = _sum(operator(ones))
        total_data = float(np.sum(data, dtype=float))
        if (total_ones > 0) and (total_data > 0):
            scale = total_data / total_ones
        else:
            scale = 1.0

        # from here on the scene is expressed in units of the reference
        # radiance.
        def forward(scene: "torch.Tensor") -> "torch.Tensor":
            return operator(scene) * scale

        def transpose(images: "torch.Tensor") -> "torch.Tensor":
            return operator_transpose(images) * scale

        variance = self._variance(
            images=images,
            data=data,
            axes=axes_images,
            scene_uniform=np.full(shape_scene, scale),
        )
        variance = np.broadcast_to(variance, data.shape)
        weight = np.divide(
            1,
            variance,
            out=np.zeros(data.shape),
            where=variance > 0,
        )

        d = _tensor(data)
        w = _tensor(weight)
        where = w > 0
        num_sample = max(int(torch.count_nonzero(where).item()), 1)
        num_sample_channel = torch.count_nonzero(where, dim=dim_sensor).clamp(min=1)

        num_voxel = int(np.prod(shape_scene))

        alpha = self.alpha
        l1_ratio = self.l1_ratio
        l1 = alpha * l1_ratio / num_voxel
        l2 = alpha * (1 - l1_ratio) / num_voxel

        if self.prior is not None:
            x0 = self._array_scene(regridder, self.prior, self._unit) / scale
            x0 = _tensor(x0)
        else:
            x0 = torch.zeros(shape_scene, device=device, dtype=dtype)

        if self.penalty_weights is not None:
            c = self._array_scene(regridder, self.penalty_weights, None)
            if np.any(c < 0):
                raise ValueError("`penalty_weights` must be nonnegative.")
            c = _tensor(c)
        else:
            c = ones

        # the linear part of the penalty
        q = l1 * c - l2 * x0

        if guess is not None:
            x = self._array_scene(regridder, guess, self._unit) / scale
            x = _tensor(x).clamp(min=0)
        else:
            # every step of the iteration is driven by the measurement, so
            # starting from the prior leaves it in place wherever the
            # measurement has nothing to say.
            x = x0.clamp(min=0)

        def gradient(image: "torch.Tensor") -> "torch.Tensor":
            # the gradient of the first term of the merit function, in terms
            # of the image predicted by the current scene
            return transpose(w * (image - d)) / num_sample

        def merit(x: "torch.Tensor", image: "torch.Tensor") -> float:
            result = _sum(w * torch.square(image - d)) / (2 * num_sample)
            result = result + l1 * _sum(c * x)
            result = result + l2 * _sum(torch.square(x - x0)) / 2
            return result

        def gap(x: "torch.Tensor", image: "torch.Tensor", f: float) -> float:
            # an upper bound on the difference between the merit function and
            # its minimum, found by evaluating the dual of the merit function.
            g = gradient(image)
            if l2 > 0:
                v = -(g + q)
                result = l2 * torch.square(x) / 2 - x * v
                result = result + torch.square(v.clamp(min=0)) / (2 * l2)
                return _sum(result)
            # without the l2 penalty the dual is constrained, and the residual
            # must be scaled until it is feasible.
            negative = g < 0
            if torch.any(negative):
                ratio = torch.where(negative, q / -g, torch.ones_like(g))
                factor = min(1.0, ratio.min().item())
            else:
                factor = 1.0
            residual = image - d
            dual = -factor * factor * _sum(w * torch.square(residual)) / 2
            dual = dual - factor * _sum(w * residual * d)
            return f - dual / num_sample

        # Since the forward model has no negative elements, the first term of
        # the merit function is bounded from above by a paraboloid which is
        # separable in the voxels, with these curvatures. Each voxel then
        # takes a step of its own size, which is much faster than a single
        # step size when the weights of the pixels span orders of magnitude.
        curvature = transpose(w * forward(ones)) / num_sample

        tolerance = 10 * torch.finfo(dtype).eps

        image = forward(x)
        f = merit(x, image)

        x_old = x
        image_old = image
        momentum = 1.0

        solutions = []
        chi2 = []
        correlation = []
        objective = []

        gap_relative = np.inf
        f_check = np.inf

        message = f"Max number of iterations ({self.num_iteration}) exceeded."
        success = False
        num_iteration = self.num_iteration

        for i in range(self.num_iteration):

            residual = d - image
            chi2_i = (w * torch.square(residual)).sum(dim_sensor) / num_sample_channel
            chi2.append(chi2_i.cpu().numpy())
            correlation.append(_pearsonr(image, residual, dim_sensor).cpu().numpy())
            objective.append(f)

            if self.intermediate:
                solutions.append(x.cpu().numpy() * scale)

            if verbose:  # pragma: nocover
                print(f"{i=}, merit={f}, chi2={chi2[~0].mean()}, {momentum=}")

            if (i % self.num_iteration_check == 0) and (i > 0):
                gap_relative = gap(x, image, f) / max(f, np.finfo(float).tiny)
                if gap_relative <= self.threshold_convergence:
                    message = (
                        f"The relative duality gap is {gap_relative:.3g}, less "
                        f"than {self.threshold_convergence}."
                    )
                    success = True
                elif (l2 == 0) and (f_check - f) <= self.threshold_convergence * f:
                    # without the l2 penalty the dual is only a tight bound
                    # very close to the minimum, so also accept a merit
                    # function which has stopped decreasing.
                    message = (
                        f"The merit function decreased by less than "
                        f"{self.threshold_convergence} of its value over the "
                        f"last {self.num_iteration_check} iterations."
                    )
                    success = True
                f_check = f
                if success:
                    num_iteration = i + 1
                    break

            momentum_new = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
            beta = (momentum - 1) / momentum_new

            # extrapolate the scene, and the image by linearity
            z = x + beta * (x - x_old)
            image_z = image + beta * (image - image_old)

            # minimize the paraboloid plus the penalty, which has a closed
            # form in every voxel
            x_new = curvature * z - (gradient(image_z) + q)
            x_new = x_new / (curvature + l2).clamp(min=1e-30)
            x_new = x_new.clamp(min=0)
            image_new = forward(x_new)
            f_new = merit(x_new, image_new)

            if f_new > f * (1 + tolerance):
                if beta > 0:
                    # the momentum overshot, restart from the current scene
                    x_old = x
                    image_old = image
                    momentum = 1.0
                else:
                    # a step without momentum can only increase the merit
                    # function through rounding errors, so be more cautious
                    curvature = 2 * curvature
                continue

            x_old = x
            image_old = image
            x = x_new
            image = image_new
            f = f_new
            momentum = momentum_new

        else:
            warnings.warn(message)

        solution = x.cpu().numpy() * scale

        if self.intermediate:
            solutions = np.stack(solutions[:num_iteration])
        else:
            solutions = solution[np.newaxis]

        axis_iteration = self.axis_iteration

        solutions = na.FunctionArray(
            inputs=instrument.coordinates_scene,
            outputs=na.ScalarArray(
                ndarray=solutions << self._unit,
                axes=(axis_iteration, *axes_scene),
            ),
        )

        axes_channel = axes_images[: len(axes_images) - 2]

        mean_chi_squared = na.ScalarArray(
            ndarray=np.stack(chi2[:num_iteration]),
            axes=(axis_iteration, *axes_channel),
        )
        correlation_residual = na.ScalarArray(
            ndarray=np.stack(correlation[:num_iteration]),
            axes=(axis_iteration, *axes_channel),
        )
        objective = na.ScalarArray(
            ndarray=np.array(objective[:num_iteration]),
            axes=(axis_iteration,),
        )

        return RegressionInversionResult(
            solutions=solutions,
            success=success,
            images=images,
            inverter=self,
            message=message,
            num_iteration=num_iteration,
            mean_chi_squared=mean_chi_squared,
            correlation_residual=correlation_residual,
            objective=objective,
            duality_gap=gap_relative,
            scale=scale << self._unit,
        )


@dataclasses.dataclass
class RidgeInverter(
    ElasticNetInverter,
):
    r"""
    Reconstruct a scene using nonnegative ridge regression,
    an :class:`ElasticNetInverter` where the penalty is purely :math:`\ell_2`.

    This spreads the radiance over all of the voxels consistent with the
    measurement, as close to :attr:`prior` as the measurement allows.
    """

    l1_ratio: float = dataclasses.field(default=0, init=False)
    r"""
    The fraction of the penalty which is :math:`\ell_1`,
    which is zero for ridge regression.
    """


@dataclasses.dataclass
class LassoInverter(
    ElasticNetInverter,
):
    r"""
    Reconstruct a scene using nonnegative lasso regression,
    an :class:`ElasticNetInverter` where the penalty is purely :math:`\ell_1`.

    This concentrates the radiance into as few voxels as possible.
    Since there are many more voxels than pixels, the reconstruction is
    generally not unique, and :attr:`prior` is ignored.
    """

    l1_ratio: float = dataclasses.field(default=1, init=False)
    r"""
    The fraction of the penalty which is :math:`\ell_1`,
    which is one for lasso regression.
    """


@dataclasses.dataclass
class RegressionInversionResult(
    IterativeInversionResult,
):
    """The results of a regression inversion attempt."""

    objective: na.ScalarArray = dataclasses.MISSING
    """
    The merit function at each iteration.

    This never increases from one iteration to the next.
    """

    duality_gap: float = dataclasses.MISSING
    """
    The last evaluation of the duality gap, as a fraction of the merit
    function.

    This is an upper bound on how far the merit function of :attr:`solution`
    is above its minimum.
    """

    scale: u.Quantity = dataclasses.MISSING
    """
    The reference radiance used to make the penalty dimensionless,
    the uniform scene which reproduces the total measured signal.
    """
