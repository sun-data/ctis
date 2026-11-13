import warnings
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import pytest
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


def _assert_same(
    result: ctis.inverters.IterativeInversionResult,
    expected: ctis.inverters.IterativeInversionResult,
) -> None:
    """
    Assert that two inversions took the same path to the same solution, to
    within the rounding of their arithmetic.
    """
    assert result.num_iteration == expected.num_iteration
    assert result.success == expected.success
    assert result.message == expected.message

    for name in ("solutions", "mean_chi_squared", "correlation_residual"):
        a = getattr(result, name)
        b = getattr(expected, name)
        if name == "solutions":
            assert np.all(a.inputs == b.inputs)
            a = a.outputs
            b = b.outputs
        assert a.shape == b.shape
        a = a.ndarray_aligned(b.axes)
        b = b.ndarray
        assert a.unit == b.unit
        assert np.allclose(
            a.value,
            b.value,
            rtol=1e-10,
            atol=1e-10 * np.abs(b.value).max(),
        )

    assert np.all(result.images == expected.images)
    assert result.inverter == expected.inverter


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

    @pytest.mark.parametrize("images", [images])
    @pytest.mark.parametrize(
        argnames="guess",
        argvalues=[
            None,
            na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit,
            na.random.uniform(
                low=1,
                high=2,
                shape_random=dict(reversed(scene.outputs.shape.items())),
                seed=0,
            )
            * scene.outputs.unit,
        ],
        ids=["no guess", "uniform guess", "guess with its axes reversed"],
    )
    def test__call__named(
        self,
        a: ctis.inverters.MartInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.ScalarArray,
    ) -> None:
        """
        The iteration on the operators of the instrument, which is the one
        used for this instrument, takes the same path to the same solution
        as the iteration on its `image` and `backproject` methods.
        """
        assert a._arrays(images.outputs, guess) is not None

        guess_original = None if guess is None else guess.copy()

        result = a(images, guess=guess)
        expected = a._call_named(images, guess, False)

        _assert_same(result, expected)

        if guess is not None:
            assert result.solution.outputs.unit == guess.unit
            assert np.all(guess == guess_original)


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


def test__call__no_runtime_warnings():
    """
    The pixels and voxels outside the field of view of each channel are
    expected to divide by zero, so an inversion must not warn about them.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = a(images)

    assert result.success


def test_plot_moments_no_runtime_warnings():
    """
    The columns of each histogram which contain no pixels are expected to
    divide by zero, so plotting the moments must not warn about them.
    """
    # Add a background so that no pixel of the scene is empty,
    # since the median of an empty pixel is undefined, and versions of
    # named-arrays before sun-data/named-arrays#255 warn about it.
    scene_background = scene + scene.max() / 100

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    result = a(instrument.image(scene_background))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig, axs = result.plot_moments(scene_background, axis="wavelength")

    plt.close(fig)


def test__call__two_channel_axes() -> None:
    """
    An instrument with more than one channel axis is inverted the same way by
    the iteration on its operators and by the iteration on its methods.
    """
    angle_two = na.ScalarArray(
        ndarray=angle.ndarray.reshape(2, 2),
        axes=("channel_a", "channel_b"),
    )
    instrument_two = dataclasses.replace(
        instrument,
        angle=angle_two,
        channel=angle_two,
        axis_channel=("channel_a", "channel_b"),
    )
    a = ctis.inverters.MartInverter(
        instrument=instrument_two,
        threshold_convergence=1e-2,
    )
    images_two = instrument_two.image(scene)

    assert a._arrays(images_two.outputs, None) is not None

    result = a(images_two)
    expected = a._call_named(images_two, None, False)

    _assert_same(result, expected)
    assert result.mean_chi_squared.shape == dict(
        iteration=result.num_iteration,
        channel_a=2,
        channel_b=2,
    )


def test__call__without_operators() -> None:
    """
    An instrument which does not provide operators is inverted by the
    iteration on its methods, with the same result.
    """
    quantum_yield = na.ScalarArray(np.ones(velocity.size - 1), axes="wavelength")
    instrument_named = dataclasses.replace(
        instrument,
        quantum_yield=quantum_yield * u.electron / u.photon,
    )
    a = ctis.inverters.MartInverter(
        instrument=instrument_named,
        threshold_convergence=1e-2,
    )

    assert a._arrays(images.outputs, None) is None

    result = a(images)
    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )(images)

    result = dataclasses.replace(result, inverter=expected.inverter)
    _assert_same(result, expected)


@pytest.mark.parametrize(
    argnames="guess",
    argvalues=[
        na.ScalarArray.ones(scene.outputs.shape),
        na.ScalarArray.ones(scene.outputs.shape) * u.photon / u.s / u.cm**2,
        na.ScalarArray.ones(dict(scene_x=64, scene_y=64)) * scene.outputs.unit,
    ],
    ids=["no unit", "another unit", "no wavelength axis"],
)
def test__call__guess_unsupported(guess: na.ScalarArray) -> None:
    """
    A guess which cannot be expressed as a plain array is left to the
    iteration on named arrays, which raises the error it always has.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    assert a._arrays(images.outputs, guess) is None

    with pytest.raises((u.UnitConversionError, ValueError)):
        a(images, guess=guess)


def test__call__images_unsupported() -> None:
    """
    Images which cannot be expressed as a plain array are left to the
    iteration on named arrays, which raises the error it always has.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )
    images_bare = images.replace(
        outputs=na.ScalarArray(
            ndarray=images.outputs.ndarray.value,
            axes=images.outputs.axes,
        ),
    )

    assert a._arrays(images_bare.outputs, None) is None

    with pytest.raises(u.UnitConversionError):
        a(images_bare)
