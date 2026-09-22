import matplotlib.pyplot as plt
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from .._iterative_test import AbstractTestAbstractIterativeInverter

velocity = na.linspace(-500, 500, axis="wavelength", num=21) * u.km / u.s

wavelength_rest = 171 * u.AA

AA = dict(unit=u.AA, equivalencies=u.doppler_optical(wavelength_rest))

wavelength = velocity.to(**AA)

position_scene = na.Cartesian2dVectorLinearSpace(
    start=-10 * u.arcsec,
    stop=10 * u.arcsec,
    axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
    num=na.Cartesian2dVectorArray(64, 64),
)

position_sensor = na.Cartesian2dVectorArray(
    x=na.arange(0, 128 + 1, axis="sensor_x") * u.pix,
    y=na.arange(0, 64 + 1, axis="sensor_y") * u.pix,
)

coordinates_scene = na.DopplerPositionalVectorArray.from_velocity(
    velocity=velocity,
    wavelength_rest=wavelength_rest,
    position=position_scene,
)
coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
    velocity=velocity,
    wavelength_rest=wavelength_rest,
    position=position_sensor,
)

scene = ctis.scenes.gaussians(coordinates_scene)

coordinates_scene.wavelength = wavelength
coordinates_sensor.wavelength = wavelength

angle = na.linspace(0, 360, num=4, axis="channel", endpoint=False) * u.deg

instrument = ctis.instruments.IdealInstrument(
    area_effective=1 * u.cm**2,
    timedelta_exposure=20 * u.s,
    plate_scale=0.4 * u.arcsec / u.pix,
    dispersion=((10 * u.km / u.s).to(**AA) - wavelength_rest) / u.pix,
    angle=angle,
    wavelength_ref=wavelength_rest,
    position_ref=na.Cartesian2dVectorArray(64, 32) * u.pix,
    coordinates_scene=coordinates_scene,
    coordinates_sensor=coordinates_sensor,
    channel=angle,
    axis_channel="channel",
    axis_wavelength="wavelength",
    axis_scene_xy=("scene_x", "scene_y"),
    axis_sensor_xy=("sensor_x", "sensor_y"),
)

images = instrument.image(scene)

inverter = ctis.inverters.MartInverter(
    instrument=instrument,
)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        ctis.inverters.MartInverter(
            instrument=instrument,
            threshold_convergence=1e-2,
        ),
        ctis.inverters.MartInverter(
            instrument=instrument,
            num_iteration=2,
            threshold_convergence=1e-2,
        ),
        ctis.inverters.MartInverter(
            instrument=instrument,
            intermediate=True,
            threshold_convergence=1e-2,
        ),
        ctis.inverters.MartInverter(
            instrument=instrument,
            regularization=1 / 4,
            threshold_convergence=1e-2,
        ),
    ],
)
class TestMartInverter(
    AbstractTestAbstractIterativeInverter,
):
    @pytest.mark.parametrize("images", [images])
    @pytest.mark.parametrize(
        argnames="guess",
        argvalues=[
            None,
            na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit,
        ],
    )
    def test__call__(
        self,
        a: ctis.inverters.AbstractInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: na.ScalarArray,
    ):
        result = super().test__call__(
            a=a,
            images=images,
            guess=guess,
        )

        fig, axs = result.plot_moments(scene, axis="wavelength")

        assert isinstance(fig, plt.Figure)
        for ax in axs:
            assert isinstance(ax, plt.Axes)


def test__call__verbose_convergence():
    """
    Verbose output must not disable the convergence check.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        num_iteration=50,
        threshold_convergence=1e-2,
    )

    result = a(images, verbose=True)

    assert result.success
    assert result.num_iteration < a.num_iteration


@pytest.mark.parametrize(
    argnames="axis_regularization,regularization",
    argvalues=[
        (None, 1 / 4),
        (("wavelength", "scene_x", "scene_y"), 1 / 4),
        (("scene_x", "wavelength"), 1 / 2),
    ],
)
def test_regularize(
    axis_regularization: None | tuple[str, ...],
    regularization: float,
):
    """
    One regularization step is a convolution with the kernel of Parker 2022
    along each regularized axis, conserves the sum along each regularized axis,
    and preserves positivity.
    """
    axis = ("wavelength", "scene_x", "scene_y")

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        regularization=regularization,
        axis_regularization=axis_regularization,
    )

    x = na.random.uniform(0, 1, shape_random=scene.outputs.shape, seed=0)
    x = x * scene.outputs.unit

    result = a.regularize(x)

    axis_regularized = a.axis_regularization_
    assert np.allclose(result.sum(axis_regularized), x.sum(axis_regularized))
    assert np.all(result >= 0)

    kernel = [regularization, 1 - 2 * regularization, regularization]

    def convolve(column: np.ndarray) -> np.ndarray:
        return np.convolve(np.pad(column, 1, mode="edge"), kernel, mode="valid")

    expected = x.ndarray_aligned(axis).value
    for ax in axis_regularized:
        expected = np.apply_along_axis(convolve, axis.index(ax), expected)

    assert np.allclose(result.ndarray_aligned(axis).value, expected)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(regularization=0.6),
        dict(regularization=-0.1),
        dict(axis_regularization="channel"),
        dict(axis_regularization=("wavelength", "sensor_x")),
    ],
)
def test_regularization_invalid(kwargs: dict):
    """
    Weights which would produce a kernel with negative weights, and axes which
    are not axes of the scene, are rejected.
    """
    with pytest.raises(ValueError):
        ctis.inverters.MartInverter(instrument=instrument, **kwargs)


def test__call__regularization():
    """A smoothness penalty produces smoother spectral line profiles."""
    axis = ("wavelength", "scene_x", "scene_y")

    def roughness(result: ctis.inverters.IterativeInversionResult) -> float:
        v = result.solution.outputs.ndarray_aligned(axis).value
        return np.mean(np.square(np.diff(v, axis=0))) / np.mean(np.square(v))

    kwargs = dict(instrument=instrument, threshold_convergence=1e-2)

    rough = ctis.inverters.MartInverter(**kwargs)(images)
    smooth = ctis.inverters.MartInverter(regularization=1 / 4, **kwargs)(images)

    assert roughness(smooth) < roughness(rough)
