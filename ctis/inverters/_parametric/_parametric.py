from typing import ClassVar, TYPE_CHECKING
import abc
import warnings
import dataclasses
import numpy as np
import astropy.units as u
import astropy.constants
import named_arrays as na
import ctis
from ..._torch import _torch
from .. import AbstractInverter, AbstractInversionResult
from ._models import AbstractSpectralModel

__all__ = [
    "AbstractParametricInverter",
    "ParametricInverter",
    "ParametricInversionResult",
]

if TYPE_CHECKING:  # pragma: nocover
    import torch


@dataclasses.dataclass
class AbstractParametricInverter(
    AbstractInverter,
):
    """
    An abstract inversion algorithm which reconstructs an observed scene by
    fitting a parameterized spectral line profile to every spatial pixel.

    Unlike :class:`~ctis.inverters.AbstractIterativeInverter`, which solves for
    the radiance in every voxel of the scene, these algorithms solve for a
    handful of parameters in every `spatial` pixel of the scene.
    For a three-parameter model this reduces the number of unknowns below the
    number of measurements, turning an underdetermined problem into an
    overdetermined one.

    The parameters of neighboring pixels are `not` independent, since each
    sensor pixel collects light from many spatial pixels.
    The fit is therefore a single optimization over every parameter of every
    pixel simultaneously.
    """

    axis_iteration: ClassVar[str] = "iteration"
    """The logical axis associated with changing iteration index."""

    @property
    @abc.abstractmethod
    def model(self) -> AbstractSpectralModel:
        """The spectral line profile fit to each spatial pixel of the scene."""


@dataclasses.dataclass
class ParametricInverter(
    AbstractParametricInverter,
):
    r"""
    Fit a parameterized spectral line profile to every spatial pixel of the
    scene using the Adam optimizer :cite:p:`Kingma2014`.

    The forward model of :attr:`instrument` is assembled into a sparse
    :class:`~ctis.Regridder`, so the gradient of the merit function with
    respect to every parameter is computed exactly by automatic
    differentiation, and the whole optimization can run on a GPU.

    The merit function is

    .. math::

        \langle \chi^2 \rangle = \left\langle
            \left( \frac{d - \hat{d}(\theta)}{\sigma} \right)^2
        \right\rangle,

    where :math:`d` are the measured electrons, :math:`\hat{d}` are the
    electrons predicted by the model, and :math:`\sigma` is estimated from the
    `measured` signal rather than the predicted signal, which avoids biasing
    the fitted radiance.

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
        # The velocity bins are chosen to be comparable to the width of the
        # spectral line, since a parametric fit cannot recover a width which
        # is much narrower than one bin.
        wavelength_rest = 630 * u.AA
        velocity = na.linspace(-250, 250, axis="wavelength", num=21) * u.km / u.s
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
        # The sensor must be large enough to hold the dispersed scene,
        # otherwise the voxels which fall off the edge are unconstrained.
        coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
            velocity=velocity,
            wavelength_rest=wavelength_rest,
            position=na.Cartesian2dVectorArray(
                x=na.arange(0, 97, axis="sensor_x") * u.pix,
                y=na.arange(0, 97, axis="sensor_y") * u.pix,
            ),
        )

        # Define an idealized CTIS instrument with four channels
        angle = na.linspace(0, 360, axis="channel", num=4, endpoint=False) * u.deg
        instrument = ctis.instruments.IdealInstrument(
            area_effective=1 * u.cm**2,
            timedelta_exposure=20 * u.s,
            plate_scale=0.625 * u.arcsec / u.pix,
            dispersion=0.021 * u.AA / u.pix,
            angle=angle,
            wavelength_ref=wavelength_rest,
            position_ref=48 * u.pix,
            coordinates_scene=coordinates_scene,
            coordinates_sensor=coordinates_sensor,
            channel=angle,
            axis_channel="channel",
            axis_wavelength="wavelength",
            axis_scene_xy=("scene_x", "scene_y"),
            axis_sensor_xy=("sensor_x", "sensor_y"),
        )

        # Simulate an observation of a test scene
        scene = ctis.scenes.gaussians(coordinates_scene)
        images = instrument.image(scene)

        # Fit a Gaussian line profile to every spatial pixel
        inverter = ctis.inverters.ParametricInverter(
            instrument=instrument,
            model=ctis.inverters.GaussianModel(
                width_thermal=11 * u.km / u.s,
                width_instrument=8 * u.km / u.s,
                velocity_max=250 * u.km / u.s,
            ),
            num_iteration=500,
        )
        result = inverter(images)

        # Plot the fitted Doppler velocity.
        # The unit is stripped from the color values since a colorbar norm
        # cannot hold an `astropy` quantity.
        velocity_fit = result.parameters["velocity"]
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            img = na.plt.pcolormesh(
                coordinates_scene.position.x,
                coordinates_scene.position.y,
                C=velocity_fit.value,
                ax=ax,
                cmap="RdBu_r",
            )
            ax.set_aspect("equal")
            plt.colorbar(
                img.ndarray.item(),
                ax=ax,
                label=f"Doppler velocity ({velocity_fit.unit:latex_inline})",
            )
    """

    instrument: ctis.instruments.AbstractLinearInstrument = dataclasses.MISSING
    """
    A model of a CTIS instrument which transforms the radiance of an observed
    scene to the electrons measured by the sensors.

    Any :class:`~ctis.instruments.AbstractLinearInstrument` is supported, since
    the forward model is rebuilt from its
    :attr:`~ctis.instruments.AbstractLinearInstrument.weights` and
    :attr:`~ctis.instruments.AbstractLinearInstrument.response`.
    """

    model: AbstractSpectralModel = dataclasses.MISSING
    """The spectral line profile fit to each spatial pixel of the scene."""

    num_iteration: int = dataclasses.field(default=500, kw_only=True)
    """The maximum number of optimizer steps to perform."""

    num_iteration_guess: int = dataclasses.field(default=100, kw_only=True)
    """
    The number of MART iterations used to compute the initial guess.

    The moments of the reconstruction found by
    :class:`~ctis.inverters.MartInverter` are used as the starting point of
    the fit.
    Since the merit function is not convex, the quality of this guess has a
    strong effect on the quality of the fit.
    """

    learning_rate: float = dataclasses.field(default=0.05, kw_only=True)
    """The initial learning rate of the Adam optimizer."""

    learning_rate_decay: float = dataclasses.field(default=0.01, kw_only=True)
    """
    The ratio of the final learning rate to :attr:`learning_rate`.

    The learning rate decays geometrically over :attr:`num_iteration` steps.
    Adam takes steps of roughly a fixed size, so an annealed learning rate is
    needed to resolve the Doppler velocity to a small fraction of a bin.
    """

    threshold_convergence: float = dataclasses.field(default=1e-6, kw_only=True)
    r"""
    The fractional decrease in :math:`\langle \chi^2 \rangle` which counts as
    an improvement.

    Together with :attr:`num_patience` this determines when the optimization
    is considered to be converged.
    """

    num_patience: int = dataclasses.field(default=50, kw_only=True)
    r"""
    The number of iterations to continue without improving
    :math:`\langle \chi^2 \rangle` before declaring convergence.

    Adam takes steps of a roughly fixed size, so the merit function is not
    monotonic and a single uphill step does not mean the fit has converged.
    """

    uncertainty: None | na.AbstractScalar = dataclasses.field(
        default=None,
        kw_only=True,
    )
    """
    The standard deviation of the measurement noise, in electrons.

    If :obj:`None` (the default) and `images` carries an uncertainty, as
    produced by ``instrument.image(uncertainty=True)``, that uncertainty is
    used.
    Otherwise the measurement is assumed to be shot-noise limited and the
    variance is estimated from the measured signal.
    """

    variance_min: float = dataclasses.field(default=1, kw_only=True)
    """
    The minimum variance, in electrons squared, assigned to a measurement.

    This prevents pixels which measured zero signal from being assigned
    infinite weight.
    """

    device: None | str = dataclasses.field(default=None, kw_only=True)
    """
    The :mod:`torch` device on which to perform the optimization.

    If :obj:`None`, a CUDA device is used if one is available.
    """

    def _validate(self) -> None:
        instrument = self.instrument

        if not isinstance(instrument, ctis.instruments.AbstractLinearInstrument):
            raise ValueError(
                f"{type(instrument)=} is not supported, the forward model must "
                f"be a `ctis.instruments.AbstractLinearInstrument`."
            )

        coordinates = instrument.coordinates_scene
        if not isinstance(coordinates, na.AbstractDopplerPositionalVectorArray):
            raise ValueError(
                "`instrument.coordinates_scene` must be a Doppler vector array "
                "since the spectral model is expressed in velocity space."
            )

    @property
    def _velocity(self) -> np.ndarray:
        """The edges of each velocity bin of the scene, in km/s."""
        instrument = self.instrument
        velocity = instrument.coordinates_scene.velocity
        velocity = velocity.ndarray_aligned((instrument.axis_wavelength,))
        return velocity.to_value(u.km / u.s)

    @property
    def _dvdl(self) -> float:
        r"""
        The derivative of Doppler velocity with respect to wavelength,
        in :math:`\text{km} \, \text{s}^{-1} \, \AA^{-1}`.
        """
        wavelength_rest = self.instrument.coordinates_scene.wavelength_rest
        result = astropy.constants.c / wavelength_rest
        return result.to_value(u.km / u.s / u.AA)

    @property
    def unit_intensity(self) -> u.UnitBase:
        """
        The unit of the fitted line radiance, integrated over the spectral line.

        This is derived from the units of
        :attr:`~ctis.instruments.AbstractLinearInstrument.response`, so an
        instrument whose forward model consumes a photon radiance yields a
        photon radiance, and one which consumes an energy radiance yields an
        energy radiance.
        """
        scale_input, scale_output = self.instrument.response

        unit = na.unit_normalized(scale_input) * na.unit_normalized(scale_output)

        result = u.electron / unit * u.AA

        # express the result in a conventional radiance unit if possible
        for candidate in (
            u.erg / (u.cm**2 * u.sr * u.s),
            u.ph / (u.cm**2 * u.sr * u.s),
        ):
            if result.is_equivalent(candidate):
                return candidate

        return result  # pragma: nocover

    def _response(
        self,
        regridder: "ctis.Regridder",
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        The two diagonal factors of the forward model, expressed as plain
        arrays laid out for :class:`~ctis.Regridder`.

        The first converts the profile returned by :attr:`model`, which is a
        radiance per unit velocity, into the quantity consumed by the weights.
        The second converts the resampled quantity into electrons.
        """
        instrument = self.instrument

        axis_wavelength = instrument.axis_wavelength
        axis_scene_xy = tuple(instrument.axis_scene_xy)

        scale_input, scale_output = instrument.response

        # The model works in velocity space while the instrument integrates
        # over wavelength, so convert the radiance density between the two,
        # then attach the unit of the intensity parameter and express the
        # product in whatever unit makes the second factor yield electrons.
        scale_input = (
            scale_input
            * self._dvdl
            * (u.km / u.s / u.AA)
            * self.unit_intensity
            / (u.km / u.s)
        )
        scale_input = scale_input.to(u.electron / na.unit_normalized(scale_output))

        axes_input = (axis_wavelength,) + axis_scene_xy
        shape_input = {ax: regridder.shape_input[ax] for ax in axes_input}
        scale_input = na.broadcast_to(na.as_named_array(scale_input), shape_input)
        scale_input = na.value(scale_input.ndarray_aligned(axes_input))

        axes_output = regridder.axes_values_output
        shape_output = {ax: regridder.shape_output[ax] for ax in axes_output}
        scale_output = na.broadcast_to(na.as_named_array(scale_output), shape_output)
        scale_output = na.value(scale_output.ndarray_aligned(axes_output))

        return scale_input, scale_output

    @property
    def unit_parameters(self) -> dict[str, u.UnitBase]:
        """
        The unit of each physical parameter of :attr:`model`.

        The unit of the line radiance is determined by the instrument, see
        :attr:`unit_intensity`, while the remaining units are determined by
        the model.
        """
        return self.model.unit(self.unit_intensity)

    def _guess(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: (
            None
            | dict[str, na.AbstractScalar]
            | na.AbstractScalar
            | na.AbstractFunctionArray
        ),
    ) -> dict[str, np.ndarray]:
        """
        Compute the starting point of the fit, as the physical value of every
        parameter in every spatial pixel.
        """
        instrument = self.instrument

        axis_wavelength = instrument.axis_wavelength
        axis_scene_xy = tuple(instrument.axis_scene_xy)

        # the physical parameters may be supplied directly, which allows a fit
        # to be warm-started from the result of a previous one.
        if isinstance(guess, dict):
            unit = self.unit_parameters
            shape = {
                ax: instrument.coordinates_scene.shape[ax] - 1 for ax in axis_scene_xy
            }
            result = dict()
            for name in self.model.parameters:
                if name not in guess:
                    raise ValueError(
                        f"`guess` is missing the parameter {name!r}, "
                        f"expected {self.model.parameters}."
                    )
                value = na.broadcast_to(na.as_named_array(guess[name]), shape)
                value = u.Quantity(value.ndarray_aligned(axis_scene_xy))
                result[name] = value.to_value(unit[name])
            return result

        if guess is None:
            inverter = ctis.inverters.MartInverter(
                instrument=instrument,
                num_iteration=self.num_iteration_guess,
                # the natural units of the backprojection differ between
                # instruments, and a photon radiance cannot be converted into
                # an energy radiance without the energy per photon, so ask for
                # the unit this fit works in.
                unit=self.unit_intensity / u.AA,
            )
            with warnings.catch_warnings():
                # the guess is deliberately stopped before convergence, and
                # MART divides by zero in voxels which received no signal.
                warnings.simplefilter("ignore", UserWarning)
                with np.errstate(invalid="ignore", divide="ignore"):
                    # MART operates on the measured values, so any uncertainty
                    # attached to the images is dropped before the guess.
                    guess = inverter(
                        images.replace(outputs=na.nominal(images.outputs)),
                    ).solution

        if isinstance(guess, na.AbstractFunctionArray):
            guess = guess.outputs

        axes = (axis_wavelength,) + axis_scene_xy

        cube = guess.ndarray_aligned(axes)
        cube = u.Quantity(cube).to_value(self.unit_intensity / u.AA)

        # the model works in velocity space, so convert the radiance density
        # from a wavelength basis into a velocity basis.
        cube = cube / self._dvdl

        cube = np.maximum(cube, 0)

        velocity = self._velocity
        width_bin = np.diff(velocity)
        center_bin = (velocity[:-1] + velocity[1:]) / 2

        shape = (-1,) + (1,) * len(axis_scene_xy)
        weight = cube * width_bin.reshape(shape)

        intensity = weight.sum(0)
        intensity_safe = np.maximum(intensity, np.finfo(float).tiny)

        center = center_bin.reshape(shape)
        mean = (weight * center).sum(0) / intensity_safe
        variance = (weight * np.square(center - mean)).sum(0) / intensity_safe

        variance_fixed = self.model._width_fixed_squared
        width_min = np.abs(width_bin).min() / 10
        width = np.sqrt(np.maximum(variance - variance_fixed, np.square(width_min)))

        intensity = np.maximum(intensity, intensity.max() / 1e6)

        return dict(
            intensity=intensity,
            velocity=mean,
            width_nonthermal=width,
        )

    def __call__(
        self,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: (
            None
            | dict[str, na.AbstractScalar]
            | na.AbstractScalar
            | na.AbstractFunctionArray
        ) = None,
        verbose: bool = False,
    ) -> "ParametricInversionResult":
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
            The starting point of the fit, given in any of three forms.

            A :class:`dict` of physical parameters, named according to
            :attr:`~ctis.inverters.AbstractSpectralModel.parameters` of
            :attr:`model`, is used directly. Each value may be a full map over
            the spatial axes of the scene, or a scalar which is broadcast over
            them. Since
            :attr:`~ctis.inverters.ParametricInversionResult.parameters` is a
            dictionary of exactly this form, the result of one fit may be used
            to warm-start another.

            A reconstructed scene, as either an
            :class:`~named_arrays.AbstractScalar` or an
            :class:`~named_arrays.AbstractFunctionArray`, is reduced to its
            moments in every spatial pixel.

            If :obj:`None` (the default), a scene is first reconstructed using
            :class:`~ctis.inverters.MartInverter`, and its moments are used.
        verbose
            Whether to print the merit function at every iteration.
        """
        torch = _torch()

        self._validate()

        instrument = self.instrument
        model = self.model

        axis_wavelength = instrument.axis_wavelength
        axis_scene_xy = tuple(instrument.axis_scene_xy)
        axis_sensor_xy = tuple(instrument.axis_sensor_xy)

        position_images = images.inputs.position
        position_sensor = instrument.coordinates_sensor.position
        if not np.all(position_images == position_sensor):
            raise ValueError(
                "`images.inputs.position` and `self.coordinates_sensor.position` "
                "are not equal."
            )

        regridder = ctis.Regridder.from_weights(
            weights=instrument.weights,
            axis_input=axis_scene_xy,
            axis_output=axis_sensor_xy,
            device=self.device,
        )

        device = regridder.device
        dtype = regridder.dtype

        axis_block = regridder.axis_block
        index_wavelength = axis_block.index(axis_wavelength)

        def _tensor(array: np.ndarray) -> "torch.Tensor":
            return torch.as_tensor(np.ascontiguousarray(array)).to(
                device=device,
                dtype=dtype,
            )

        velocity = _tensor(self._velocity)

        scale_input, scale_output = self._response(regridder)
        scale_input = _tensor(scale_input)
        scale_output = _tensor(scale_output)

        axes_data = tuple(ax for ax in axis_block if ax != axis_wavelength)
        axes_data = axes_data + axis_sensor_xy

        outputs = images.outputs

        data = na.nominal(outputs)
        data = u.Quantity(data.ndarray_aligned(axes_data)).to_value(u.electron)

        if self.uncertainty is not None:
            variance = np.square(
                u.Quantity(
                    na.as_named_array(self.uncertainty).ndarray_aligned(axes_data)
                ).to_value(u.electron)
            )
        elif isinstance(outputs, na.AbstractUncertainScalarArray):
            # the instrument was asked for its own uncertainty model
            width = outputs.width.ndarray_aligned(axes_data)
            variance = np.square(u.Quantity(width).to_value(u.electron))
        else:
            # shot-noise limited, estimated from the measured signal rather
            # than the predicted signal, which would bias the radiance low.
            variance = np.maximum(data, 0)

        variance = np.maximum(variance, self.variance_min)

        data = _tensor(data)
        weight = _tensor(1 / np.sqrt(variance))

        shape_cube = regridder.shape_values_input

        def forward(parameters: "torch.Tensor") -> "torch.Tensor":
            cube = model(parameters, velocity) * scale_input
            for i, ax in enumerate(axis_block):
                if ax != axis_wavelength:
                    cube = cube.unsqueeze(i)
            cube = cube.expand(shape_cube)
            result = regridder(cube) * scale_output
            return result.sum(index_wavelength)

        parameters = self._guess(images, guess)
        parameters = model.guess(**{k: _tensor(v) for k, v in parameters.items()})
        parameters = parameters.detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam([parameters], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.learning_rate_decay ** (1 / max(self.num_iteration - 1, 1)),
        )

        mean_chi_squared = []
        merit_best = np.inf
        merit_reference = np.inf
        iteration_reference = 0
        parameters_best = parameters.detach().clone()
        message = f"Max number of iterations ({self.num_iteration}) exceeded."
        success = False
        num_iteration = self.num_iteration

        for i in range(self.num_iteration):

            optimizer.zero_grad(set_to_none=True)

            residual = (data - forward(parameters)) * weight
            merit = torch.mean(torch.square(residual))

            merit.backward()
            optimizer.step()
            scheduler.step()

            merit = merit.item()
            mean_chi_squared.append(merit)

            # Adam takes steps of a roughly fixed size, so the merit function
            # is not monotonic. Keep the best solution seen so far instead of
            # whichever one the last step happened to land on.
            if merit < merit_best:
                merit_best = merit
                parameters_best = parameters.detach().clone()

            if verbose:  # pragma: nocover
                print(f"{i=}, merit={merit}")

            # only a decrease larger than the threshold resets the patience
            if merit < merit_reference * (1 - self.threshold_convergence):
                merit_reference = merit
                iteration_reference = i

            if (i - iteration_reference) >= self.num_patience:
                message = (
                    f"The merit function did not decrease by more than "
                    f"{self.threshold_convergence} over the last "
                    f"{self.num_patience} iterations."
                )
                success = True
                num_iteration = i + 1
                break

        else:
            warnings.warn(message)

        parameters = parameters_best

        with torch.no_grad():
            physical = model.physical(parameters)
            profile = model(parameters, velocity)

        unit = self.unit_parameters
        parameters_result = {
            k: na.ScalarArray(
                ndarray=physical[k].detach().cpu().numpy() << unit[k],
                axes=axis_scene_xy,
            )
            for k in physical
        }

        # convert the radiance density from a velocity basis back into the
        # wavelength basis expected by the rest of the package.
        outputs = profile.detach().cpu().numpy() * self._dvdl
        outputs = na.ScalarArray(
            ndarray=outputs << (self.unit_intensity / u.AA),
            axes=(axis_wavelength,) + axis_scene_xy,
        )

        solution = na.FunctionArray(
            inputs=instrument.coordinates_scene,
            outputs=outputs,
        )

        mean_chi_squared = na.ScalarArray(
            ndarray=np.array(mean_chi_squared),
            axes=(self.axis_iteration,),
        )

        return ParametricInversionResult(
            solution=solution,
            parameters=parameters_result,
            success=success,
            images=images,
            inverter=self,
            message=message,
            num_iteration=num_iteration,
            mean_chi_squared=mean_chi_squared,
        )


@dataclasses.dataclass
class ParametricInversionResult(
    AbstractInversionResult,
):
    """The results of a parametric inversion attempt."""

    solution: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray] = (
        dataclasses.MISSING
    )
    """
    The reconstructed scene found by the inversion.

    This is the spectral line profile evaluated using :attr:`parameters`.
    """

    parameters: dict[str, na.ScalarArray] = dataclasses.MISSING
    """
    The fitted value of each physical parameter in every spatial pixel.

    The keys are the names of the parameters of
    :attr:`~ctis.inverters.AbstractParametricInverter.model`.
    """

    success: bool = dataclasses.MISSING
    """A boolean flag indicating whether the inversion was successful."""

    images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray] = (
        dataclasses.MISSING
    )
    """The observed images on which the inversion was performed."""

    inverter: "ctis.inverters.AbstractInverter" = dataclasses.MISSING
    """The inversion algorithm instance that produced these results."""

    message: str = dataclasses.MISSING
    """Any message from the inversion routine concerning the results."""

    num_iteration: int = dataclasses.MISSING
    """The number of iterations performed by the inverter."""

    mean_chi_squared: na.ScalarArray = dataclasses.MISSING
    """
    The mean chi squared statistic after each iteration.

    This is not monotonic, since Adam takes steps of a roughly fixed size.
    :attr:`parameters` is the best solution found over all iterations, not the
    solution found by the last one.
    """

    @property
    def iteration(self) -> na.ScalarArray:
        """The iteration value for each iteration."""
        return na.arange(
            start=0,
            stop=self.num_iteration,
            axis=self.inverter.axis_iteration,
        )
