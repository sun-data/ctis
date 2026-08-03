import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from .._inverters_test import AbstractTestAbstractInverter

torch = pytest.importorskip("torch")


wavelength_rest = 630 * u.AA

#: velocity bins comparable to the width of the line, and a sensor large
#: enough to hold the dispersed scene.
velocity = na.linspace(-200, 200, axis="wavelength", num=17) * u.km / u.s

coordinates_scene = na.DopplerPositionalVectorArray.from_velocity(
    velocity=velocity,
    wavelength_rest=wavelength_rest,
    position=na.Cartesian2dVectorLinearSpace(
        start=-5 * u.arcsec,
        stop=+5 * u.arcsec,
        axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
        num=17,
    ),
)

coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
    velocity=velocity,
    wavelength_rest=wavelength_rest,
    position=na.Cartesian2dVectorArray(
        x=na.arange(0, 65, axis="sensor_x") * u.pix,
        y=na.arange(0, 65, axis="sensor_y") * u.pix,
    ),
)

angle = na.linspace(0, 360, num=4, axis="channel", endpoint=False) * u.deg

instrument = ctis.instruments.IdealInstrument(
    area_effective=1 * u.cm**2,
    timedelta_exposure=20 * u.s,
    plate_scale=0.625 * u.arcsec / u.pix,
    dispersion=0.021 * u.AA / u.pix,
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

model = ctis.inverters.GaussianModel(
    width_thermal=11 * u.km / u.s,
    width_instrument=8 * u.km / u.s,
    velocity_max=200 * u.km / u.s,
)


def _truth() -> dict[str, np.ndarray]:
    """A random set of physical parameters for every spatial pixel."""
    rng = np.random.default_rng(3)
    num = 16
    return dict(
        intensity=2000.0 + 3000.0 * rng.random((num, num)),
        velocity=60.0 * (2 * rng.random((num, num)) - 1),
        width_nonthermal=15.0 + 25.0 * rng.random((num, num)),
    )


def _scene(
    truth: dict[str, np.ndarray],
) -> na.FunctionArray[na.DopplerPositionalVectorArray, na.ScalarArray]:
    """Evaluate the spectral model to produce a scene it can represent exactly."""
    inverter = ctis.inverters.ParametricInverter(instrument=instrument, model=model)
    unit = inverter.unit_intensity / u.AA

    parameters = model.guess(
        **{k: torch.as_tensor(v, dtype=torch.float32) for k, v in truth.items()}
    )
    profile = model(
        parameters,
        torch.as_tensor(inverter._velocity, dtype=torch.float32),
    )
    profile = profile.numpy() * inverter._dvdl

    return na.FunctionArray(
        inputs=coordinates_scene,
        outputs=na.ScalarArray(
            ndarray=profile << unit,
            axes=("wavelength", "scene_x", "scene_y"),
        ),
    )


truth = _truth()
scene = _scene(truth)
images = instrument.image(scene, noise=False)


class AbstractTestAbstractParametricInverter(
    AbstractTestAbstractInverter,
):

    def test_model(self, a: ctis.inverters.AbstractParametricInverter):
        result = a.model
        assert isinstance(result, ctis.inverters.AbstractSpectralModel)

    def test__call__(
        self,
        a: ctis.inverters.AbstractParametricInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        **kwargs,
    ) -> ctis.inverters.ParametricInversionResult:

        result = super().test__call__(a=a, images=images, **kwargs)

        assert result.num_iteration > 0
        assert result.mean_chi_squared.shape[a.axis_iteration] == result.num_iteration
        assert result.iteration.size == result.num_iteration

        # the reported solution is the best one seen, so the merit can
        # never end up worse than where it started
        merit = result.mean_chi_squared.ndarray
        assert merit.min() <= merit[0]

        assert isinstance(result.parameters, dict)
        for name in a.model.parameters:
            assert name in result.parameters
            parameter = result.parameters[name]
            assert isinstance(parameter, na.ScalarArray)
            assert parameter.shape == {ax: 16 for ax in a.instrument.axis_scene_xy}
            assert np.all(np.isfinite(parameter.ndarray))

        assert np.all(result.solution.outputs >= 0)

        return result


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        ctis.inverters.ParametricInverter(
            instrument=instrument,
            model=model,
            num_iteration=400,
        ),
        ctis.inverters.ParametricInverter(
            instrument=instrument,
            model=model,
            num_iteration=400,
            num_iteration_guess=10,
            learning_rate=0.1,
            device="cpu",
        ),
    ],
)
class TestParametricInverter(
    AbstractTestAbstractParametricInverter,
):
    @pytest.mark.parametrize("images", [images])
    @pytest.mark.parametrize(
        argnames="guess",
        argvalues=[
            None,
            scene,
        ],
    )
    def test__call__(
        self,
        a: ctis.inverters.ParametricInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.AbstractFunctionArray,
    ):
        with pytest.warns(UserWarning):
            return super().test__call__(a=a, images=images, guess=guess)


def test__call__recovery():
    """
    A scene which the model can represent exactly must be recovered to high
    accuracy from noiseless images.
    """
    inverter = ctis.inverters.ParametricInverter(
        instrument=instrument,
        model=model,
        num_iteration=2000,
    )

    result = inverter(images)

    assert result.success
    assert result.num_iteration < inverter.num_iteration

    axis = ("scene_x", "scene_y")

    for name in model.parameters:
        expected = truth[name]
        got = result.parameters[name].ndarray_aligned(axis).value
        r = np.corrcoef(got.ravel(), expected.ravel())[0, 1]
        assert r > 0.95, f"{name} was recovered with a correlation of only {r}"

    merit = result.mean_chi_squared.ndarray
    assert merit.min() < merit[0] / 100


def test__call__invalid_position():
    inverter = ctis.inverters.ParametricInverter(
        instrument=instrument,
        model=model,
        num_iteration=2,
    )
    with pytest.raises(ValueError):
        inverter(images.replace(inputs=coordinates_scene))


def test__call__invalid_instrument():
    """The forward model must be a linear instrument."""
    inverter = ctis.inverters.ParametricInverter(
        instrument=object(),
        model=model,
    )
    with pytest.raises(ValueError):
        inverter(images)


def test__call__invalid_coordinates():
    """The spectral model requires scene coordinates expressed in velocity."""
    inverter = ctis.inverters.ParametricInverter(
        instrument=dataclasses.replace(
            instrument,
            coordinates_scene=na.SpectralPositionalVectorArray(
                wavelength=coordinates_scene.wavelength,
                position=coordinates_scene.position,
            ),
        ),
        model=model,
    )
    with pytest.raises(ValueError):
        inverter(images)


def _instrument_optika() -> ctis.instruments.OptikaInstrument:
    """A CTIS instrument whose forward model is an `optika` linear system."""
    import optika

    channel = na.linspace(0, 360, axis="channel", num=3, endpoint=False) * u.deg
    system = optika.systems.LinearSystem(
        area_effective=optika.radiometry.InterpolatedEffectiveAreaModel(
            wavelength=na.linspace(400, 700, axis="wavelength", num=10) * u.nm,
            area=na.linspace(1, 2, axis="wavelength", num=10) * u.cm**2,
            axis_wavelength="wavelength",
        ),
        distortion=optika.distortion.SimpleDistortionModel(
            plate_scale=0.75 * u.arcsec / u.pix,
            dispersion=3.75 * u.nm / u.pix,
            angle=channel,
            reference=na.SpectralPositionalVectorArray(
                wavelength=550 * u.nm,
                position=na.Cartesian2dVectorArray(16, 16) * u.pix,
            ),
        ),
        sensor=optika.sensors.ImagingSensor(
            width_pixel=15 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("sensor_x", "sensor_y"),
            timedelta_exposure=1 * u.s,
            num_pixel=na.Cartesian2dVectorArray(32, 32),
        ),
    )
    return ctis.instruments.OptikaInstrument(
        system=system,
        coordinates_scene=na.DopplerPositionalVectorArray.from_velocity(
            velocity=na.linspace(-4000, 4000, axis="wavelength", num=7) * u.km / u.s,
            wavelength_rest=550 * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-6 * u.arcsec,
                stop=+6 * u.arcsec,
                axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
                num=17,
            ),
        ),
        channel=channel,
        axis_channel="channel",
        axis_wavelength="wavelength",
        axis_scene_xy=("scene_x", "scene_y"),
    )


@pytest.mark.parametrize(
    argnames="a,b",
    argvalues=[
        (instrument, model),
        (
            _instrument_optika(),
            ctis.inverters.GaussianModel(
                width_thermal=200 * u.km / u.s,
                width_instrument=100 * u.km / u.s,
                velocity_max=4000 * u.km / u.s,
            ),
        ),
    ],
)
def test_forward_matches_instrument(
    a: ctis.instruments.AbstractLinearInstrument,
    b: ctis.inverters.AbstractSpectralModel,
):
    """
    The differentiable forward model assembled from `weights` and `response`
    must reproduce `instrument.image` for every kind of linear instrument.
    """
    inverter = ctis.inverters.ParametricInverter(instrument=a, model=b)

    axis_wavelength = a.axis_wavelength
    axis_scene_xy = tuple(a.axis_scene_xy)
    num = tuple(a.coordinates_scene.shape[ax] - 1 for ax in axis_scene_xy)

    rng = np.random.default_rng(1)
    parameters = b.guess(
        intensity=torch.as_tensor(1 + rng.random(num), dtype=torch.float32),
        velocity=torch.as_tensor(
            b.velocity_max.to_value(u.km / u.s) / 8 * (2 * rng.random(num) - 1),
            dtype=torch.float32,
        ),
        width_nonthermal=torch.as_tensor(
            b.width_thermal.to_value(u.km / u.s) + rng.random(num),
            dtype=torch.float32,
        ),
    )
    velocity = torch.as_tensor(inverter._velocity, dtype=torch.float32)
    profile = b(parameters, velocity)

    scene = na.FunctionArray(
        inputs=a.coordinates_scene,
        outputs=na.ScalarArray(
            ndarray=(profile.numpy() * inverter._dvdl)
            << (inverter.unit_intensity / u.AA),
            axes=(axis_wavelength,) + axis_scene_xy,
        ),
    )
    expected = a.image(scene, noise=False)

    regridder = ctis.Regridder.from_weights(
        weights=a.weights,
        axis_input=axis_scene_xy,
        axis_output=a.axis_sensor_xy,
        device="cpu",
    )
    axis_block = regridder.axis_block
    scale_input, scale_output = inverter._response(regridder)

    cube = profile * torch.as_tensor(scale_input, dtype=torch.float32)
    for i, ax in enumerate(axis_block):
        if ax != axis_wavelength:
            cube = cube.unsqueeze(i)
    cube = cube.expand(regridder.shape_values_input)
    result = regridder(cube) * torch.as_tensor(scale_output, dtype=torch.float32)
    result = result.sum(axis_block.index(axis_wavelength)).detach().numpy()

    axes = tuple(ax for ax in axis_block if ax != axis_wavelength)
    axes = axes + tuple(a.axis_sensor_xy)
    expected = u.Quantity(expected.outputs.ndarray_aligned(axes)).to_value(u.electron)

    # the absolute tolerance is scaled to the signal so that pixels which
    # received no light are not compared to float32 rounding noise
    assert np.allclose(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5 * expected.max(),
    )


def test__call__uncertainty_explicit():
    """An uncertainty may be supplied directly instead of being estimated."""
    uncertainty = np.sqrt(np.abs(images.outputs.value)) * u.electron
    uncertainty = uncertainty + 1 * u.electron

    inverter = ctis.inverters.ParametricInverter(
        instrument=instrument,
        model=model,
        num_iteration=20,
        num_iteration_guess=5,
        uncertainty=uncertainty,
    )

    with pytest.warns(UserWarning):
        result = inverter(images)

    assert np.all(np.isfinite(result.mean_chi_squared.ndarray))


def test__call__uncertainty_from_images():
    """The uncertainty attached to the images by the instrument is used."""
    images_uncertain = instrument.image(scene, noise=False, uncertainty=True)

    assert isinstance(images_uncertain.outputs, na.AbstractUncertainScalarArray)

    inverter = ctis.inverters.ParametricInverter(
        instrument=instrument,
        model=model,
        num_iteration=20,
        num_iteration_guess=5,
    )

    with pytest.warns(UserWarning):
        result = inverter(images_uncertain)

    assert np.all(np.isfinite(result.mean_chi_squared.ndarray))


def test__call__optika():
    """
    The whole inversion must run against an `OptikaInstrument`, whose
    backprojection is naturally expressed in photon rather than energy units.
    """
    a = _instrument_optika()
    b = ctis.inverters.GaussianModel(
        width_thermal=200 * u.km / u.s,
        width_instrument=100 * u.km / u.s,
        velocity_max=4000 * u.km / u.s,
    )

    inverter = ctis.inverters.ParametricInverter(
        instrument=a,
        model=b,
        num_iteration=50,
        num_iteration_guess=5,
    )

    axis_scene_xy = tuple(a.axis_scene_xy)
    num = tuple(a.coordinates_scene.shape[ax] - 1 for ax in axis_scene_xy)
    num_wavelength = a.coordinates_scene.shape[a.axis_wavelength] - 1

    rng = np.random.default_rng(0)
    scene_optika = na.FunctionArray(
        inputs=a.coordinates_scene,
        outputs=na.ScalarArray(
            ndarray=(1 + rng.random((num_wavelength,) + num))
            << (inverter.unit_intensity / u.AA),
            axes=(a.axis_wavelength,) + axis_scene_xy,
        ),
    )

    with pytest.warns(UserWarning):
        result = inverter(a.image(scene_optika, noise=False))

    assert np.all(np.isfinite(result.mean_chi_squared.ndarray))
    for name in b.parameters:
        assert np.all(np.isfinite(result.parameters[name].ndarray))
