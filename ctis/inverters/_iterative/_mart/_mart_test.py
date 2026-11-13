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


def _uses_operators(
    a: ctis.inverters.MartInverter,
    images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    guess: None | na.ScalarArray = None,
) -> bool:
    """
    Whether an inversion iterates on the operators of the instrument, instead
    of its `image` and `backproject` methods.
    """
    shape_frames = a._shape_frames(images.outputs)
    return a._arrays(images.outputs, guess, shape_frames) is not None


@dataclasses.dataclass
class _IdealInstrumentWeights(
    ctis.instruments.IdealInstrument,
):
    """
    An ideal instrument which does not provide its matrices, so that it is
    imaged, backprojected and inverted through its weights, as named arrays,
    one wavelength at a time.
    """

    def _operators(self, dtype: type = np.float64) -> None:
        return None


def _reference(
    a: ctis.inverters.MartInverter,
    images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    guess: None | na.ScalarArray = None,
) -> ctis.inverters.IterativeInversionResult:
    """
    The inversion which `a` is expected to reproduce: the iteration on the
    `image` and `backproject` methods of the same instrument without its
    matrices, which shares none of the code that the matrices run on.

    The result is given the inverter `a`, so that the two can be compared.
    """
    instrument = a.instrument
    kwargs = {
        field.name: getattr(instrument, field.name)
        for field in dataclasses.fields(instrument)
        if field.init
    }
    reference = dataclasses.replace(
        a,
        instrument=_IdealInstrumentWeights(**kwargs),
    )

    assert not _uses_operators(reference, images, guess)

    result = reference(images, guess=guess)

    return dataclasses.replace(result, inverter=a)


def _assert_same(
    result: ctis.inverters.IterativeInversionResult,
    expected: ctis.inverters.IterativeInversionResult,
) -> None:
    """
    Assert that two inversions took the same path to the same solution, to
    within the rounding of their arithmetic.
    """
    assert np.all(result.num_iteration == expected.num_iteration)
    assert np.all(result.success == expected.success)
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
            atol=1e-10 * np.nanmax(np.abs(b.value)),
            equal_nan=True,
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
        as the iteration on the `image` and `backproject` methods of the
        same instrument without its matrices.
        """
        assert _uses_operators(a, images, guess)

        guess_original = None if guess is None else guess.copy()

        result = a(images, guess=guess)
        expected = _reference(a, images, guess)

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
    the iteration on its operators and by the iteration on the methods of
    the same instrument without its matrices.
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

    assert _uses_operators(a, images_two)

    result = a(images_two)
    expected = _reference(a, images_two)

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

    assert not _uses_operators(a, images)

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

    assert not _uses_operators(a, images, guess)

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

    assert not _uses_operators(a, images_bare)

    with pytest.raises(u.UnitConversionError):
        a(images_bare)


def _images_frames(
    shape_frames: dict[str, int],
) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
    """
    The images of several frames, each of the scene at a different
    brightness, so that the frames converge after different numbers of
    iterations.

    The scene has a background, without which the inversion of every frame
    stops at its second iteration.
    """
    scene_background = scene + scene.max() / 100
    expected = instrument.image(scene_background, noise=False).outputs
    values = expected.ndarray.value

    num_frame = int(np.prod(list(shape_frames.values())))
    rng = np.random.default_rng(0)
    frames = [rng.poisson(k * values) for k in np.geomspace(0.2, 5, num_frame)]
    frames = np.stack(frames, axis=~0)
    frames = frames.reshape(values.shape + tuple(shape_frames.values()))

    return na.FunctionArray(
        inputs=images.inputs,
        outputs=na.ScalarArray(
            ndarray=frames * u.electron,
            axes=(*expected.axes, *shape_frames),
        ),
    )


def _assert_frame(
    result: ctis.inverters.IterativeInversionResult,
    index: dict[str, int],
    expected: ctis.inverters.IterativeInversionResult,
) -> None:
    """
    Assert that one frame of an inversion of several frames is the inversion
    of that frame alone, bit for bit.

    The iterations which follow the last of the frame keep its solution and
    have no merit.
    """
    axis_iteration = result.inverter.axis_iteration
    num = expected.num_iteration

    assert result.num_iteration[index] == num
    assert result.success[index] == expected.success

    solutions = result.solutions.outputs[index]
    solutions_expected = expected.solutions.outputs
    num_solutions = solutions_expected.shape[axis_iteration]

    assert solutions.unit == solutions_expected.unit
    assert np.all(
        solutions[{axis_iteration: slice(num_solutions)}] == solutions_expected
    )
    assert np.all(
        solutions[{axis_iteration: slice(num_solutions, None)}]
        == solutions_expected[{axis_iteration: ~0}]
    )
    assert np.all(result.solution.outputs[index] == expected.solution.outputs)

    for name in ("mean_chi_squared", "correlation_residual"):
        merit = getattr(result, name)[index]
        merit_expected = getattr(expected, name)
        assert np.all(merit[{axis_iteration: slice(num)}] == merit_expected)
        assert np.all(np.isnan(merit[{axis_iteration: slice(num, None)}]))


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(),
        dict(intermediate=True),
        dict(num_iteration=3),
        dict(num_iteration=3, intermediate=True),
    ],
    ids=["final solutions", "every solution", "too few iterations", "both"],
)
@pytest.mark.parametrize(
    argnames="shape_frames",
    argvalues=[
        dict(time=3),
        dict(time=2, repeat=2),
    ],
    ids=["one axis", "two axes"],
)
def test__call__frames(kwargs: dict, shape_frames: dict[str, int]) -> None:
    """
    Each frame of several is reconstructed as it would be alone, bit for bit,
    and stops at its own iteration.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
        **kwargs,
    )
    images_frames = _images_frames(shape_frames)

    assert _uses_operators(a, images_frames)

    result = a(images_frames)

    assert result.solutions.outputs.shape == dict(
        iteration=result.iteration.size if a.intermediate else 1,
        **scene.outputs.shape,
        **shape_frames,
    )
    assert result.mean_chi_squared.shape == dict(
        iteration=result.iteration.size,
        **shape_frames,
        channel=4,
    )
    assert result.iteration.size == result.num_iteration.max()
    assert result.num_iteration.shape == shape_frames
    assert result.success.shape == shape_frames
    assert np.all(result.images == images_frames)

    for index in na.ndindex(shape_frames):
        _assert_frame(result, index, a(images_frames[index]))

    if "num_iteration" in kwargs:
        assert not np.any(result.success)
        assert np.all(result.num_iteration == a.num_iteration)
        assert "exceeded" in result.message
    else:
        # the frames stop at iterations of their own
        assert np.all(result.success)
        assert result.num_iteration.min() < result.num_iteration.max()
        assert "Achieved" in result.message


def test__call__frames_unconverged() -> None:
    """
    If only some of the frames converge within the iterations allowed, the
    others stop at the last, as they would alone.
    """
    shape_frames = dict(time=3)
    images_frames = _images_frames(shape_frames)

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )
    num_iteration = a(images_frames).num_iteration

    # as many iterations as the first frame to converge needs, and no more
    a = dataclasses.replace(a, num_iteration=int(num_iteration.ndarray.min()))

    with pytest.warns(UserWarning, match="exceeded"):
        result = a(images_frames)

    assert np.any(result.success)
    assert not np.all(result.success)
    assert np.all(result.success == (num_iteration == a.num_iteration))
    assert "exceeded" in result.message

    for index in na.ndindex(shape_frames):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = a(images_frames[index])
        _assert_frame(result, index, expected)


@pytest.mark.parametrize(
    argnames="guess",
    argvalues=[
        na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit,
        na.random.uniform(
            low=1,
            high=2,
            shape_random=dict(time=3) | scene.outputs.shape,
            seed=0,
        )
        * scene.outputs.unit,
    ],
    ids=["one guess", "a guess for each frame"],
)
def test__call__frames_guess(guess: na.ScalarArray) -> None:
    """
    A guess is shared by the frames, unless it has their axes, and is not
    overwritten.
    """
    shape_frames = dict(time=3)
    images_frames = _images_frames(shape_frames)

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    assert _uses_operators(a, images_frames, guess)

    guess_original = guess.copy()

    result = a(images_frames, guess=guess)

    assert result.solution.outputs.unit == guess.unit
    assert np.all(guess == guess_original)

    for index in na.ndindex(shape_frames):
        _assert_frame(result, index, a(images_frames[index], guess=guess[index]))


@pytest.mark.parametrize("intermediate", [False, True])
def test__call__frames_without_operators(intermediate: bool) -> None:
    """
    An instrument which does not provide operators has its frames
    reconstructed one at a time, with the same result.
    """
    shape_frames = dict(time=3)
    images_frames = _images_frames(shape_frames)

    quantum_yield = na.ScalarArray(np.ones(velocity.size - 1), axes="wavelength")
    instrument_named = dataclasses.replace(
        instrument,
        quantum_yield=quantum_yield * u.electron / u.photon,
    )
    a = ctis.inverters.MartInverter(
        instrument=instrument_named,
        threshold_convergence=1e-2,
        intermediate=intermediate,
    )

    assert not _uses_operators(a, images_frames)

    result = a(images_frames)
    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
        intermediate=intermediate,
    )(images_frames)

    result = dataclasses.replace(result, inverter=expected.inverter)
    _assert_same(result, expected)


@pytest.mark.parametrize(
    argnames="dtype",
    argvalues=[np.float32, "float32", np.dtype("float32")],
    ids=["type", "name", "dtype"],
)
def test_dtype(dtype: type | str | np.dtype) -> None:
    """
    In single precision, each frame stops at the iteration it stops at in
    double precision, at the same solution to within the precision, and is
    still reconstructed as it would be alone, bit for bit.
    """
    shape_frames = dict(time=3)
    images_frames = _images_frames(shape_frames)

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
        dtype=dtype,
    )

    assert _uses_operators(a, images_frames)

    result = a(images_frames)
    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )(images_frames)

    assert result.solutions.outputs.ndarray.dtype == np.float32
    assert result.mean_chi_squared.ndarray.dtype == np.float64
    assert result.correlation_residual.ndarray.dtype == np.float64

    assert np.all(result.num_iteration == expected.num_iteration)
    assert np.all(result.success == expected.success)

    for name in ("solutions", "mean_chi_squared", "correlation_residual"):
        value = getattr(result, name)
        value_expected = getattr(expected, name)
        if name == "solutions":
            value = value.outputs
            value_expected = value_expected.outputs
        assert value.unit == value_expected.unit
        value = value.ndarray_aligned(value_expected.axes).value
        value_expected = value_expected.ndarray.value
        assert np.allclose(
            value,
            value_expected,
            rtol=1e-5,
            atol=1e-5 * np.nanmax(np.abs(value_expected)),
            equal_nan=True,
        )

    for index in na.ndindex(shape_frames):
        _assert_frame(result, index, a(images_frames[index]))


def test_dtype_guess() -> None:
    """
    In single precision, the solution is in the units of the guess, and in
    single precision.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
        dtype=np.float32,
    )
    guess = na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit

    result = a(images, guess=guess)

    assert result.solution.outputs.unit == guess.unit
    assert result.solution.outputs.ndarray.dtype == np.float32
    assert guess.ndarray.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.complex128])
def test_dtype_invalid(dtype: type) -> None:
    with pytest.raises(ValueError, match="dtype"):
        ctis.inverters.MartInverter(instrument=instrument, dtype=dtype)


def test_dtype_without_operators() -> None:
    """
    An instrument which does not provide operators cannot be inverted in
    single precision.
    """
    quantum_yield = na.ScalarArray(np.ones(velocity.size - 1), axes="wavelength")
    instrument_named = dataclasses.replace(
        instrument,
        quantum_yield=quantum_yield * u.electron / u.photon,
    )
    a = ctis.inverters.MartInverter(
        instrument=instrument_named,
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="dtype"):
        a(images)
