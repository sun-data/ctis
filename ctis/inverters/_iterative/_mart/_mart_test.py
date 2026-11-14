import warnings
import weakref
import tracemalloc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import pytest
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
import ctis
from ctis.instruments._operators import Operators
from ctis.instruments._instruments_test import _instrument_optika
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
    return a._arrays(images.outputs, guess) is not None


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


@dataclasses.dataclass
class _OptikaInstrumentNamed(
    ctis.instruments.OptikaInstrument,
):
    """
    An instrument on a linear system of :mod:`optika` which does not provide
    its matrices, so that it is inverted through its methods, as named
    arrays.
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
    cls = _IdealInstrumentWeights
    if isinstance(instrument, ctis.instruments.OptikaInstrument):
        cls = _OptikaInstrumentNamed

    reference = dataclasses.replace(
        a,
        instrument=cls(**kwargs),
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


def test_dtype_fallback() -> None:
    """
    Single precision is refused whenever the iteration cannot run on the
    operators, and not only when the instrument does not provide them.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        dtype=np.float32,
    )
    shape_guess = scene.outputs.shape | dict(time=1)
    guess = na.ScalarArray.ones(shape_guess) * scene.outputs.unit

    assert not _uses_operators(a, images, guess)

    with pytest.raises(ValueError, match="only available on the sparse matrices"):
        a(images, guess=guess)


def _assert_close(a: na.AbstractScalar, b: na.AbstractScalar) -> None:
    """
    Assert that two arrays have the same axes, the same unit, and the same
    values to within the rounding of their arithmetic.
    """
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


def _instrument_time(num: int) -> ctis.instruments.IdealInstrument:
    """
    The instrument with an exposure time for each element of an axis of its
    own, `time`, each the exposure time of the instrument.
    """
    return dataclasses.replace(
        instrument,
        timedelta_exposure=na.ScalarArray(np.full(num, 20) * u.s, axes="time"),
    )


def test__call__instrument_axis() -> None:
    """
    An axis of the images which the instrument itself varies along is not an
    axis of frames: the images are inverted along it in one iteration on the
    methods of the instrument, with the result they have without it.
    """
    a = ctis.inverters.MartInverter(
        instrument=_instrument_time(1),
        threshold_convergence=1e-2,
    )
    images_time = images.replace(outputs=images.outputs.add_axes("time"))

    assert not _uses_operators(a, images_time)

    result = a(images_time)
    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )(images)

    assert isinstance(result.num_iteration, int)
    assert result.num_iteration == expected.num_iteration
    assert result.success == expected.success
    assert result.solution.outputs.shape["time"] == 1
    _assert_close(result.solution.outputs[dict(time=0)], expected.solution.outputs)


def test__call__instrument_axis_frames() -> None:
    """
    Beside an axis which the instrument itself varies along, the other axes
    of the images are still those of frames, which are reconstructed one at
    a time with the axis of the instrument carried along.
    """
    shape_frames = dict(repeat=2)
    images_frames = _images_frames(shape_frames)

    a = ctis.inverters.MartInverter(
        instrument=_instrument_time(1),
        threshold_convergence=1e-2,
    )
    images_time = images_frames.replace(
        outputs=images_frames.outputs.add_axes("time"),
    )

    result = a(images_time)
    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )(images_frames)

    assert result.num_iteration.shape == shape_frames
    assert np.all(result.num_iteration == expected.num_iteration)
    assert np.all(result.success == expected.success)
    assert result.solution.outputs.shape["time"] == 1
    _assert_close(result.solution.outputs[dict(time=0)], expected.solution.outputs)


def test__call__instrument_axis_many() -> None:
    """
    An instrument which varies along an axis of the images with more than
    one element is refused before any work, since the iteration on its
    methods cannot stop its elements one at a time.
    """
    instrument_time = _instrument_time(2)
    images_time = instrument_time.image(scene, noise=False)

    assert images_time.outputs.shape["time"] == 2

    a = ctis.inverters.MartInverter(instrument=instrument_time)

    with pytest.raises(ValueError, match="not those of independent frames"):
        a(images_time)


def test__call__axis_scene() -> None:
    """
    Images with an axis named after an axis of the scene are not a stack of
    frames: they are left to the iteration on the methods of the instrument,
    which inverts them as it always has.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )
    images_wavelength = images.replace(outputs=images.outputs.add_axes("wavelength"))

    assert _uses_operators(a, images)
    assert not _uses_operators(a, images_wavelength)

    result = a(images_wavelength)
    expected = a(images)

    assert result.num_iteration == expected.num_iteration
    assert result.success == expected.success
    _assert_close(result.solution.outputs, expected.solution.outputs)


def test__call__axis_iteration() -> None:
    """Images with the axis of the iterations of the result are refused."""
    images_iteration = images.replace(
        outputs=images.outputs.add_axes(inverter.axis_iteration),
    )

    with pytest.raises(ValueError, match="the axis of the iterations"):
        inverter(images_iteration)


def test__call__num_iteration_large() -> None:
    """
    The memory of an inversion does not grow with the number of iterations
    it is allowed, only with the number it runs.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
        num_iteration=10**6,
    )

    assert _uses_operators(a, images)

    # the inversion it is compared with also derives the matrix of the
    # backprojection, if no inversion has yet, which takes memory of its own
    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )(images)

    # the merit of a million iterations would take 64 MB
    tracemalloc.start()
    result = a(images)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 32 * 2**20

    assert result.num_iteration == expected.num_iteration
    assert result.mean_chi_squared.shape[a.axis_iteration] == result.num_iteration
    _assert_close(result.mean_chi_squared, expected.mean_chi_squared)


@pytest.mark.parametrize("name", ["mean_chi_squared", "correlation_residual"])
def test__call__merit_replaced(name: str) -> None:
    """
    An inverter which replaces a merit function is left to it: the iteration
    runs on the methods of the instrument, which call the function.
    """

    def doubled(
        self: ctis.inverters.MartInverter,
        *args: na.ScalarArray,
        **kwargs: na.ScalarArray,
    ) -> na.ScalarArray:
        return 2 * getattr(ctis.inverters.MartInverter, name)(self, *args, **kwargs)

    cls = type("Doubled", (ctis.inverters.MartInverter,), {name: doubled})

    a = cls(instrument=instrument, threshold_convergence=1e-2)

    assert not _uses_operators(a, images)

    result = a(images)
    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )(images)

    index = {a.axis_iteration: 0}

    _assert_close(getattr(result, name)[index], 2 * getattr(expected, name)[index])


def test__call__buffer_released(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The array which every iteration overwrites with the photons leaving
    every voxel belongs to the inversion, and nothing keeps it once the
    inversion has returned.
    """
    buffers = []
    image = Operators.image

    def image_recorded(
        self: Operators,
        scene: np.ndarray,
        buffer: None | np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        buffers.append(weakref.ref(buffer))
        return image(self, scene, buffer=buffer)

    monkeypatch.setattr(Operators, "image", image_recorded)

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    result = a(images)

    assert len(buffers) == result.num_iteration
    assert all(buffer() is None for buffer in buffers)


def _images_optika(
    a: ctis.instruments.OptikaInstrument,
) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
    """
    The images, without noise, of a scene which fills the field of view of
    an instrument on a linear system of :mod:`optika`.
    """
    operators = a._operators()
    scene = na.random.uniform(
        low=1,
        high=2,
        shape_random=operators.shape_scene,
        seed=0,
    )
    return a.image(scene * 1e3 * operators.unit_scene, noise=False)


@pytest.mark.parametrize(
    argnames="vignetted,guessed",
    argvalues=[(True, False), (False, True)],
    ids=["vignetted, without a guess", "plain, with a guess"],
)
def test__call__optika(vignetted: bool, guessed: bool) -> None:
    """
    An instrument on a linear system of :mod:`optika` is inverted on its
    matrices, with the result of the iteration on its methods: with
    vignetting and a field stop, from the first guess of the inverter, and
    without them, from a guess.

    Without vignetting, the first guess of the inverter fills the sensor
    evenly, and the correlation of even images with anything is rounding
    error, which the two iterations do not share.
    """
    a = ctis.inverters.MartInverter(
        instrument=_instrument_optika(vignetted=vignetted),
        num_iteration=6,
        threshold_convergence=-np.inf,
    )
    images_optika = _images_optika(a.instrument)

    guess = None
    if guessed:
        operators = a.instrument._operators()
        guess = na.random.uniform(
            low=1,
            high=2,
            shape_random=operators.shape_scene,
            seed=1,
        )
        guess = guess * 1e3 * operators.unit_scene

    assert _uses_operators(a, images_optika, guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_optika, guess=guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = _reference(a, images_optika, guess)

    assert result.num_iteration == 6
    _assert_same(result, expected)


def test__call__optika_line() -> None:
    """
    Across one spectral line, where the operators take one variance-to-mean
    ratio for the sensor, the solution after a given number of iterations is
    that of the iteration on the methods of the instrument, and the merit is
    within the tolerance of its own.
    """
    instrument_line = _instrument_optika(
        wavelength=[62.9, 63.1] * u.nm,
        material=optika.sensors.materials.e2v_ccd97(),
    )
    a = ctis.inverters.MartInverter(
        instrument=instrument_line,
        num_iteration=4,
        threshold_convergence=-np.inf,
    )
    images_line = _images_optika(instrument_line)

    assert _uses_operators(a, images_line)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_line)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = _reference(a, images_line)

    assert result.num_iteration == expected.num_iteration == 4
    _assert_close(result.solution.outputs, expected.solution.outputs)

    ratio = result.mean_chi_squared / expected.mean_chi_squared
    error = np.abs(na.value(ratio).ndarray - 1)

    assert error.max() < instrument_line._tolerance_vmr
    assert error.max() > 1e-8


def test_dtype_optika() -> None:
    """
    An instrument on a linear system of :mod:`optika` is inverted in single
    precision too, to the solution of double precision to within the
    precision.
    """
    instrument_optika = _instrument_optika()
    images_optika = _images_optika(instrument_optika)

    kwargs = dict(
        instrument=instrument_optika,
        num_iteration=6,
        threshold_convergence=-np.inf,
    )

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = ctis.inverters.MartInverter(dtype=np.float32, **kwargs)(images_optika)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = ctis.inverters.MartInverter(**kwargs)(images_optika)

    assert result.solution.outputs.ndarray.dtype == np.float32
    assert result.num_iteration == expected.num_iteration

    value = result.solution.outputs.ndarray_aligned(expected.solution.outputs.axes)
    value_expected = expected.solution.outputs.ndarray

    assert value.unit == value_expected.unit
    assert np.allclose(
        value.value,
        value_expected.value,
        rtol=1e-5,
        atol=1e-5 * np.abs(value_expected.value).max(),
    )


def _instrument_lines() -> ctis.instruments.IdealInstrument:
    """
    The instrument with a scene of two spectral lines, which overlap on its
    sensors and each have a quantum yield of their own.
    """
    wavelength_lines = na.ScalarArray([171.0, 171.05] * u.AA, axes="line")
    wavelength_lines = wavelength_lines * (1 + velocity / astropy.constants.c)
    wavelength_lines = wavelength_lines.to(u.AA)

    quantum_yield = na.ScalarArray(np.array([3.75, 2.99]), axes="line")

    return dataclasses.replace(
        instrument,
        coordinates_scene=na.SpectralPositionalVectorArray(
            wavelength=wavelength_lines,
            position=position_scene,
        ),
        coordinates_sensor=na.SpectralPositionalVectorArray(
            wavelength=wavelength_lines,
            position=position_sensor,
        ),
        axis_line="line",
        quantum_yield=quantum_yield * u.electron / u.photon,
    )


def _scene_lines() -> na.ScalarArray:
    """A scene of two spectral lines, the second fainter than the first."""
    return na.stack(
        arrays=[
            scene.outputs,
            scene.outputs[dict(wavelength=slice(None, None, -1))] / 2,
        ],
        axis="line",
    )


def test__call__lines() -> None:
    """
    A scene of several spectral lines is reconstructed on the matrices, with
    the result of the iteration on the methods of the instrument, and the
    solution has the axis of the lines.
    """
    instrument_lines = _instrument_lines()
    scene_lines = _scene_lines()
    images_lines = instrument_lines.image(scene_lines, noise=False)

    assert images_lines.outputs.shape == images.outputs.shape

    a = ctis.inverters.MartInverter(
        instrument=instrument_lines,
        num_iteration=6,
        threshold_convergence=-np.inf,
    )

    assert _uses_operators(a, images_lines)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_lines)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = _reference(a, images_lines)

    assert result.solution.outputs.shape == scene_lines.shape
    assert result.num_iteration == 6
    _assert_same(result, expected)


def test__call__lines_frames() -> None:
    """
    Frames of a scene of several spectral lines are reconstructed together,
    each as it would be alone, bit for bit, in single precision too.
    """
    instrument_lines = _instrument_lines()
    scene_lines = _scene_lines() + _scene_lines().max() / 100
    brightness = na.ScalarArray(np.array([0.5, 2.0]), axes="time")

    values = instrument_lines.image(scene_lines * brightness, noise=False).outputs
    rng = np.random.default_rng(0)
    values = values.replace(ndarray=rng.poisson(values.ndarray.value) * u.electron)
    images_frames = na.FunctionArray(inputs=images.inputs, outputs=values)

    shape_frames = dict(time=2)

    for dtype in (np.float64, np.float32):
        a = ctis.inverters.MartInverter(
            instrument=instrument_lines,
            threshold_convergence=1e-2,
            dtype=dtype,
        )

        assert _uses_operators(a, images_frames)

        result = a(images_frames)

        assert result.solution.outputs.shape == scene_lines.shape | shape_frames
        assert result.solution.outputs.ndarray.dtype == dtype

        for index in na.ndindex(shape_frames):
            _assert_frame(result, index, a(images_frames[index]))
