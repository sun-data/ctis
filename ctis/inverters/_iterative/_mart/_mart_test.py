import re
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
from ctis.instruments._instruments_test import (
    _count_checks,
    _instrument_optika,
    _instrument_optika_lines,
    _MaterialPhotons,
)
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
    return not isinstance(a._arrays(images.outputs, guess), str)


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

    # the reference is the slower iteration by design, which the inverter
    # warns of
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This inversion runs on the")
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

        fig, axs = result.plot_moments_histogram(scene, axis="wavelength")

        assert isinstance(fig, plt.Figure)
        for ax in axs.flat:
            assert isinstance(ax, plt.Axes)

        plt.close(fig)

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


def test_plot_moments_histogram_no_runtime_warnings():
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
        fig, axs = result.plot_moments_histogram(
            truth=scene_background,
            axis="wavelength",
        )

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
    iteration on its methods, with the same result, and with a warning
    which gives the reason and points to the call of the inverter.
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

    with pytest.warns(UserWarning, match="slower than on its sparse") as record:
        result = a(images)

    (warning,) = record
    reason = "since its quantum yield has the axis 'wavelength'"

    assert reason in str(warning.message)
    assert warning.filename == __file__

    expected = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )(images)

    result = dataclasses.replace(result, inverter=expected.inverter)
    _assert_same(result, expected)


def test__call__operators_silent() -> None:
    """
    An inversion which runs on the matrices of the instrument does not
    warn that it runs on its methods.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    assert _uses_operators(a, images)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="This inversion runs on the")
        result = a(images)

    assert result.success


def test__call__operators_undefined() -> None:
    """
    An instrument which defines no operators is inverted by the iteration
    on its methods, with a warning which says that it defines none.
    """
    kwargs = {
        field.name: getattr(instrument, field.name)
        for field in dataclasses.fields(instrument)
        if field.init
    }
    a = ctis.inverters.MartInverter(
        instrument=_IdealInstrumentWeights(**kwargs),
        num_iteration=2,
        threshold_convergence=-np.inf,
    )

    reason = "the instrument provides none, since it does not define any."

    with pytest.warns(UserWarning) as record:
        a(images)

    messages = [str(warning.message) for warning in record]

    assert len(messages) == 2
    assert messages[0].endswith(reason)
    assert messages[1].startswith("Max number of iterations")


@pytest.mark.parametrize(
    argnames="guess,reason",
    argvalues=[
        (
            na.ScalarArray.ones(scene.outputs.shape),
            "the guess is dimensionless, which cannot be converted to",
        ),
        (
            na.ScalarArray.ones(scene.outputs.shape) * u.photon / u.s / u.cm**2,
            "the guess is in units of `ph",
        ),
    ],
    ids=["no unit", "another unit"],
)
def test__call__guess_unsupported(guess: na.ScalarArray, reason: str) -> None:
    """
    A guess which cannot be expressed as a plain array is left to the
    iteration on named arrays, with a warning which says why, and that
    iteration raises the error it always has.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    assert not _uses_operators(a, images, guess)

    with pytest.warns(UserWarning, match=re.escape(reason)):
        with pytest.raises(u.UnitConversionError):
            a(images, guess=guess)


@pytest.mark.parametrize(
    argnames="shape_frames,shape_guess",
    argvalues=[
        (dict(), scene.outputs.shape | dict(scene_x=64)),
        (dict(), scene.outputs.shape | dict(time=3)),
        (dict(), scene.outputs.shape | dict(time=1)),
        (dict(time=3), scene.outputs.shape | dict(time=2)),
        (dict(time=3), scene.outputs.shape | dict(time=4)),
        (dict(time=3), scene.outputs.shape | dict(repeat=3)),
    ],
    ids=[
        "another length along an axis of the scene",
        "an axis which one frame does not have",
        "an axis of one element which one frame does not have",
        "fewer elements than the frames",
        "more elements than the frames",
        "an axis which the frames do not have",
    ],
)
def test__call__guess_axes(
    monkeypatch: pytest.MonkeyPatch,
    shape_frames: dict[str, int],
    shape_guess: dict[str, int],
) -> None:
    """
    A guess which cannot be broadcast to the scene of every frame is
    refused, on the operators of the instrument and on its methods, before
    either images anything: one with an axis which is neither of the scene
    nor of the frames, or with neither one element nor all of them along
    one which is.
    """
    images_frames = _images_frames(shape_frames) if shape_frames else images
    guess = na.ScalarArray.ones(shape_guess) * scene.outputs.unit

    def image(
        self: ctis.instruments.IdealInstrument | Operators,
        scene: na.AbstractScalar | np.ndarray,
        **kwargs: bool | np.ndarray,
    ) -> None:
        raise AssertionError("The guess was imaged.")  # pragma: nocover

    monkeypatch.setattr(ctis.instruments.IdealInstrument, "image", image)
    monkeypatch.setattr(Operators, "image", image)

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    with pytest.raises(ValueError, match="Every axis of the guess"):
        a(images_frames, guess=guess)

    # the iteration on the methods of the instrument refuses it the same way
    with pytest.raises(ValueError, match="Every axis of the guess"):
        _reference(a, images_frames, guess)


@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize(
    argnames="shape_frames,shape_guess",
    argvalues=[
        (dict(), dict()),
        (
            dict(),
            {ax: scene.outputs.shape[ax] for ax in ("scene_x", "scene_y")},
        ),
        (dict(), scene.outputs.shape | dict(wavelength=1)),
        (dict(time=3), dict(time=3)),
    ],
    ids=[
        "one value",
        "without the axis of the wavelengths",
        "one element along the axis of the wavelengths",
        "one value for each frame",
    ],
)
def test__call__guess_broadcast(
    shape_frames: dict[str, int],
    shape_guess: dict[str, int],
    intermediate: bool,
) -> None:
    """
    A guess is broadcast to the scene of every frame: it is the same along
    an axis of the scene or of the frames which it does not have, or has
    one element along.
    It is inverted bit for bit as the guess with every axis is, and the
    methods of the instrument give the same, whether or not the solution of
    every iteration is kept, and it is not overwritten.
    """
    images_frames = _images_frames(shape_frames) if shape_frames else images
    shape = scene.outputs.shape | shape_frames

    guess = na.random.uniform(low=1, high=2, shape_random=shape_guess, seed=0)
    guess = guess * scene.outputs.unit
    guess_original = guess.copy()

    # the same guess with every axis, which a product with ones gives
    guess_whole = guess * na.ScalarArray.ones(shape)

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
        intermediate=intermediate,
    )

    assert guess.shape == shape_guess
    assert guess_whole.shape == shape
    assert _uses_operators(a, images_frames, guess)

    result = a(images_frames, guess=guess)
    expected = a(images_frames, guess=guess_whole)

    assert np.all(result.solutions.outputs == expected.solutions.outputs)
    _assert_same(result, expected)

    # the iteration on the methods of the instrument broadcasts it the same way
    _assert_same(_reference(a, images_frames, guess), expected)

    assert np.all(guess == guess_original)


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

    with pytest.warns(UserWarning, match="the array of the images is dimension"):
        with pytest.raises(u.UnitConversionError):
            a(images_bare)


@pytest.mark.parametrize(
    argnames="shape_frames",
    argvalues=[dict(), dict(time=2)],
    ids=["one frame", "two frames"],
)
def test__call__images_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    shape_frames: dict[str, int],
) -> None:
    """
    Images which are given with the positions of the sensor of the
    instrument, to within rounding, are inverted exactly as images on the
    positions themselves are, on the operators of the instrument and on its
    methods, which check them once for each inversion, and images on other
    positions are refused.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )
    images_frames = _images_frames(shape_frames) if shape_frames else images
    inputs = images_frames.inputs
    position = inputs.position.explicit

    images_rounded = images_frames.replace(
        inputs=inputs.replace(position=position * (1 + 1e-12)),
    )

    assert not np.all(images_rounded.inputs.position == inputs.position)

    expected = a(images_frames)
    expected_reference = _reference(a, images_frames)

    count = _count_checks(monkeypatch)

    result = a(images_rounded)
    result_reference = _reference(a, images_rounded)

    assert count["num"] == 2

    for r, e in ((result, expected), (result_reference, expected_reference)):
        assert np.all(r.num_iteration == e.num_iteration)
        assert np.all(r.solutions.outputs == e.solutions.outputs)

    images_other = images_frames.replace(
        inputs=inputs.replace(position=2 * position),
    )

    with pytest.raises(ValueError, match="`images.inputs`"):
        a(images_other)


def test__call__images_without_coordinates() -> None:
    """
    Images which are given without their coordinates are refused, with an
    error which says so.
    """
    with pytest.raises(TypeError, match="with their coordinates"):
        inverter(images.outputs)


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
        na.random.uniform(
            low=1,
            high=2,
            shape_random=dict(time=1) | scene.outputs.shape,
            seed=0,
        )
        * scene.outputs.unit,
    ],
    ids=[
        "one guess",
        "a guess for each frame",
        "one guess along the axis of the frames",
    ],
)
def test__call__frames_guess(guess: na.ScalarArray) -> None:
    """
    A guess is shared by the frames, unless it has their axes, and is not
    overwritten.
    A guess with one element along an axis of the frames is that of every
    frame along it, on the operators of the instrument and on its methods.
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

    # the guess of each frame, which a product with ones gives
    guess_frames = guess * na.ScalarArray.ones(shape_frames)

    for index in na.ndindex(shape_frames):
        expected = a(images_frames[index], guess=guess_frames[index])
        _assert_frame(result, index, expected)

    _assert_same(_reference(a, images_frames, guess), result)
    assert np.all(guess == guess_original)


@pytest.mark.parametrize(
    argnames="shape_frames",
    argvalues=[
        dict(),
        dict(time=3),
    ],
    ids=["one frame", "a guess for each frame"],
)
def test__call__guess_function(shape_frames: dict[str, int]) -> None:
    """
    The solution of an inversion can be given whole, with its coordinates,
    as the guess of another.
    It is inverted exactly as its values are, on the operators of the
    instrument and on its methods alike, to a solution which is a function
    of the coordinates of the scene alone, and it is not overwritten.
    """
    images_frames = _images_frames(shape_frames) if shape_frames else images

    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    guess = a(images_frames).solution
    values = guess.outputs.copy()

    assert isinstance(guess, na.FunctionArray)
    assert _uses_operators(a, images_frames, guess.outputs)

    result = a(images_frames, guess=guess)
    expected = a(images_frames, guess=guess.outputs)

    assert isinstance(result.solution.outputs, na.ScalarArray)
    assert np.all(result.solutions.outputs == expected.solutions.outputs)
    _assert_same(result, expected)

    # the iteration on the methods of the instrument takes it the same way
    reference = _reference(a, images_frames, guess)

    assert isinstance(reference.solution.outputs, na.ScalarArray)
    _assert_same(reference, expected)

    assert np.all(guess.outputs == values)


def test__call__guess_function_rounding() -> None:
    """
    A guess which is given with the grid of the scene of the instrument,
    in another kind of vector and through another unit and back, is
    inverted exactly as its values are.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )
    inputs = instrument.coordinates_scene
    guess = na.FunctionArray(
        inputs=na.SpectralPositionalVectorArray(
            wavelength=inputs.wavelength.to(u.nm),
            position=inputs.position.to(u.rad).to(u.arcsec),
        ),
        outputs=na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit,
    )

    assert not np.all(guess.inputs.position == inputs.position)

    result = a(images, guess=guess)
    expected = a(images, guess=guess.outputs)

    assert np.all(result.solutions.outputs == expected.solutions.outputs)
    _assert_same(result, expected)


def test__call__guess_function_coordinates() -> None:
    """
    A guess which is given with coordinates other than those of the scene
    of the instrument is refused, as the instrument refuses a scene on
    another grid.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )
    inputs = instrument.coordinates_scene
    guess = na.FunctionArray(
        inputs=inputs.replace(position=2 * inputs.position),
        outputs=na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit,
    )

    with pytest.raises(ValueError, match="`guess.inputs`"):
        a(images, guess=guess)


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

    with pytest.warns(UserWarning, match="its quantum yield has the axis 'wav"):
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


@pytest.mark.parametrize(
    argnames="guess",
    argvalues=[
        na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit,
        na.FunctionArray(
            inputs=instrument.coordinates_scene,
            outputs=na.ScalarArray.ones(scene.outputs.shape) * scene.outputs.unit,
        ),
    ],
    ids=["values alone", "with its coordinates"],
)
def test_dtype_guess(guess: na.ScalarArray | na.FunctionArray) -> None:
    """
    In single precision, the solution is in the units of the guess, and in
    single precision.
    A guess which is given with its coordinates is inverted on the
    operators of the instrument too, which alone provide single precision.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
        dtype=np.float32,
    )
    values = guess.outputs if isinstance(guess, na.FunctionArray) else guess

    result = a(images, guess=guess)

    assert result.solution.outputs.unit == values.unit
    assert result.solution.outputs.ndarray.dtype == np.float32
    assert values.ndarray.dtype == np.float64


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

    with pytest.raises(ValueError, match="its quantum yield has the axis 'wav"):
        a(images)


@pytest.mark.parametrize(
    argnames="images_fallback,guess,reason",
    argvalues=[
        (
            images.replace(outputs=images.outputs.add_axes("wavelength")),
            None,
            "the images have the axes ('wavelength',), which are axes of the",
        ),
        (
            images,
            na.ScalarArray.ones(scene.outputs.shape),
            "the guess is dimensionless, which cannot be converted to",
        ),
        (
            images,
            na.ScalarArray.ones(scene.outputs.shape) * u.photon / u.s / u.cm**2,
            "the guess is in units of `ph",
        ),
    ],
    ids=[
        "images with an axis of the scene",
        "a guess without a unit",
        "a guess in another unit",
    ],
)
def test_dtype_fallback(
    images_fallback: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    guess: None | na.ScalarArray,
    reason: str,
) -> None:
    """
    Single precision is refused whenever the iteration cannot run on the
    operators, and not only when the instrument does not provide them:
    here because of the images, or because of the guess, and the error
    says which.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        dtype=np.float32,
    )

    assert not _uses_operators(a, images_fallback, guess)

    with pytest.raises(ValueError, match="only available on the sparse") as info:
        a(images_fallback, guess=guess)

    assert reason in str(info.value)


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

    with pytest.warns(UserWarning, match="its voxels has the axis 'time'"):
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

    with pytest.warns(UserWarning, match="its voxels has the axis 'time'"):
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


@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize(
    argnames="shape_frames",
    argvalues=[dict(), dict(repeat=2)],
    ids=["one frame", "a guess for each frame"],
)
def test__call__instrument_axis_guess(
    shape_frames: dict[str, int],
    intermediate: bool,
) -> None:
    """
    A guess need not have an axis which the instrument itself varies along:
    it is broadcast along it, as it is along an axis of the scene, and the
    images are inverted to the solution they have without that axis,
    whether or not the solution of every iteration is kept.
    """
    images_frames = _images_frames(shape_frames) if shape_frames else images
    images_time = images_frames.replace(
        outputs=images_frames.outputs.add_axes("time"),
    )

    guess = na.random.uniform(
        low=1,
        high=2,
        shape_random=scene.outputs.shape | shape_frames,
        seed=0,
    )
    guess = guess * scene.outputs.unit

    kwargs = dict(threshold_convergence=1e-2, intermediate=intermediate)
    a = ctis.inverters.MartInverter(instrument=_instrument_time(1), **kwargs)

    assert not _uses_operators(a, images_time, guess)

    with pytest.warns(UserWarning, match="its voxels has the axis 'time'"):
        result = a(images_time, guess=guess)
    expected = ctis.inverters.MartInverter(instrument=instrument, **kwargs)(
        images_frames,
        guess=guess,
    )

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

    reason = "the images have the axes ('wavelength',), which are axes of the"

    with pytest.warns(UserWarning, match=re.escape(reason)):
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


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[True, False],
    ids=["with its coordinates", "values alone"],
)
def test__call__guess_axis_iteration(function: bool) -> None:
    """
    A guess with the axis of the iterations of the result is refused, as
    the `solutions` of an earlier inversion are, which are one letter from
    its `solution`.
    """
    a = ctis.inverters.MartInverter(
        instrument=instrument,
        threshold_convergence=1e-2,
    )

    guess = a(images).solutions
    if not function:
        guess = guess.outputs

    with pytest.raises(ValueError, match="the guess has an axis"):
        a(images, guess=guess)


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

    with pytest.warns(UserWarning, match="a `Doubled`, replaces a merit"):
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


def test_gamma_default() -> None:
    """The learning rate is one unless told otherwise."""
    assert ctis.inverters.MartInverter(instrument=instrument).gamma == 1


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


unit_radiance = u.erg / (u.cm**2 * u.s * u.sr * u.AA)
"""A unit of energy for a scene, which is not that of any of the matrices."""


def _guess_optika(
    a: ctis.instruments.OptikaInstrument,
    unit: None | u.UnitBase = None,
) -> na.ScalarArray:
    """
    A guess at the scene of an instrument on a linear system of
    :mod:`optika`, in the unit of the scene of its matrices, which is in
    photons, or in `unit`, a unit of energy.

    The guess in units of energy is the one in photons times the energy of
    a photon at the mean wavelength of the scene, so that the two are about
    as bright.
    """
    operators = a._operators()
    guess = na.random.uniform(
        low=1,
        high=2,
        shape_random=operators.shape_scene,
        seed=1,
    )
    guess = guess * 1e3 * operators.unit_scene

    if unit is not None:
        wavelength = a.coordinates_scene.wavelength.mean()
        energy = astropy.constants.h * astropy.constants.c / wavelength / u.photon
        guess = (guess * energy).to(unit)

    return guess


@pytest.mark.parametrize(
    argnames="kwargs,guessed",
    argvalues=[
        (dict(vignetted=True), False),
        (dict(), True),
        (
            dict(
                vignetted=True,
                axis_field=("sensor_x", "sensor_y"),
                stretched=True,
            ),
            False,
        ),
    ],
    ids=[
        "vignetted, without a guess",
        "plain, with a guess",
        "scene named like the pixels",
    ],
)
def test__call__optika(kwargs: dict, guessed: bool) -> None:
    """
    An instrument on a linear system of :mod:`optika` is inverted on its
    matrices, with the result of the iteration on its methods: with
    vignetting and a field stop, from the first guess of the inverter,
    without them, from a guess, and with a scene on the grid of another
    instrument, whose spatial axes are named like the pixels of the sensor
    and whose cells differ in size.

    Without vignetting, the first guess of the inverter fills the sensor
    evenly, and the correlation of even images with anything is rounding
    error, which the two iterations do not share.
    """
    a = ctis.inverters.MartInverter(
        instrument=_instrument_optika(**kwargs),
        num_iteration=6,
        threshold_convergence=-np.inf,
    )
    images_optika = _images_optika(a.instrument)

    guess = None
    if guessed:
        guess = _guess_optika(a.instrument)

    assert _uses_operators(a, images_optika, guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_optika, guess=guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = _reference(a, images_optika, guess)

    assert result.num_iteration == 6
    _assert_same(result, expected)


@pytest.mark.parametrize(
    argnames="kwargs,unit,intermediate",
    argvalues=[
        (dict(), None, False),
        (
            dict(
                material=optika.sensors.materials.e2v_ccd97(),
                lines=[0, 60] * u.nm,
            ),
            unit_radiance,
            True,
        ),
    ],
    ids=[
        "one band, in the unit of the matrices times an energy",
        "two lines, in another unit of energy, every solution",
    ],
)
def test__call__optika_energy(
    kwargs: dict,
    unit: None | u.UnitBase,
    intermediate: bool,
) -> None:
    """
    A guess in units of energy is inverted on the matrices of an instrument
    on a linear system of :mod:`optika`, which take a scene in photons, with
    the result of the iteration on the methods of the instrument, whose
    sensor divides the scene by the energy of a photon at each wavelength.
    Every solution is in the units of the guess, which is not overwritten.

    A guess in the unit of the scene of the matrices times that of the
    energy of a photon is the one whose values need no conversion, so that
    they are the ones which would be overwritten if any were.

    The two lines are sixty nanometres apart, so the photons of one have a
    tenth more energy than those of the other.
    """
    a = ctis.inverters.MartInverter(
        instrument=_instrument_optika(**kwargs),
        num_iteration=6,
        threshold_convergence=-np.inf,
        intermediate=intermediate,
    )
    images_optika = _images_optika(a.instrument)

    operators = a.instrument._operators()

    if unit is None:
        unit = operators.unit_scene * operators.unit_energy

    guess = _guess_optika(a.instrument, unit=unit)
    guess_original = guess.copy()

    # the energy of a photon changes across the scene, so one energy for
    # every voxel would not give the same solution
    assert not guess.unit.is_equivalent(operators.unit_scene)
    assert np.ptp(operators.energy) > 0.01 * operators.energy.min()
    assert _uses_operators(a, images_optika, guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_optika, guess=guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = _reference(a, images_optika, guess)

    num_solution = result.solutions.outputs.shape[a.axis_iteration]

    assert result.num_iteration == 6
    assert (num_solution > 1) == intermediate
    assert result.solutions.outputs.unit == guess.unit
    assert np.all(guess == guess_original)
    _assert_same(result, expected)


def test__call__optika_energy_frames() -> None:
    """
    Frames are reconstructed together from a guess in units of energy for
    each of them, each as it would be alone, bit for bit.
    """
    instrument_optika = _instrument_optika()
    a = ctis.inverters.MartInverter(
        instrument=instrument_optika,
        num_iteration=4,
        threshold_convergence=-np.inf,
    )

    shape_frames = dict(time=2)
    brightness = na.ScalarArray(np.array([0.5, 2.0]), axes="time")

    images_optika = _images_optika(instrument_optika)
    images_frames = na.FunctionArray(
        inputs=images_optika.inputs,
        outputs=images_optika.outputs * brightness,
    )
    guess = _guess_optika(instrument_optika, unit=unit_radiance) * brightness

    assert _uses_operators(a, images_frames, guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_frames, guess=guess)

    assert result.solution.outputs.unit == guess.unit
    assert result.solution.outputs.shape == guess.shape

    for index in na.ndindex(shape_frames):
        with pytest.warns(UserWarning, match="Max number of iterations"):
            expected = a(images_frames[index], guess=guess[index])
        _assert_frame(result, index, expected)


@pytest.mark.parametrize(
    argnames="material,unit,reason",
    argvalues=[
        (None, u.electron, "the guess is in units of `electron`, which cannot"),
        (
            _MaterialPhotons(),
            unit_radiance,
            "the guess is in units of energy, which the matrices of the",
        ),
    ],
    ids=[
        "neither in photons nor in units of energy",
        "in units of energy, for a sensor which takes photons alone",
    ],
)
def test__call__optika_guess_unsupported(
    material: None | optika.sensors.materials.AbstractSensorMaterial,
    unit: u.UnitBase,
    reason: str,
) -> None:
    """
    A guess which the matrices cannot take is left to the iteration on the
    methods of the instrument, with a warning which says why, and that
    iteration raises the error it always has: a
    guess which is neither in photons nor in units of energy, and a guess
    in units of energy for an instrument whose sensor takes light in
    photons alone.
    """
    a = ctis.inverters.MartInverter(
        instrument=_instrument_optika(material=material),
    )
    images_optika = _images_optika(a.instrument)

    shape_scene = a.instrument._operators().shape_scene
    guess = na.ScalarArray.ones(shape_scene) * unit

    assert not _uses_operators(a, images_optika, guess)

    with pytest.warns(UserWarning, match=re.escape(reason)):
        with pytest.raises(u.UnitConversionError):
            a(images_optika, guess=guess)


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


def test__call__optika_assigned() -> None:
    """
    An instrument whose system is assigned after an inversion is inverted on
    its matrices again, with the response of its new sensor folded into
    them: the result is that of the iteration on its methods, which read the
    system every time they are called.
    """
    instrument_optika = _instrument_optika(vignetted=True)
    a = ctis.inverters.MartInverter(
        instrument=instrument_optika,
        num_iteration=4,
        threshold_convergence=-np.inf,
    )

    with pytest.warns(UserWarning, match="Max number of iterations"):
        first = a(_images_optika(instrument_optika))

    system = instrument_optika.system
    sensor = dataclasses.replace(
        system.sensor,
        timedelta_exposure=10 * u.s,
        read_noise=7 * u.electron,
    )
    instrument_optika.system = dataclasses.replace(system, sensor=sensor)

    images_optika = _images_optika(instrument_optika)

    assert _uses_operators(a, images_optika)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_optika)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = _reference(a, images_optika)

    assert result.num_iteration == 4
    assert np.all(result.mean_chi_squared != first.mean_chi_squared)
    _assert_same(result, expected)


@pytest.mark.parametrize(
    argnames="energy",
    argvalues=[False, True],
    ids=["without a guess", "with a guess in units of energy"],
)
def test_dtype_optika(energy: bool) -> None:
    """
    An instrument on a linear system of :mod:`optika` is inverted in single
    precision too, to the solution of double precision to within the
    precision, from the first guess of the inverter and from a guess in
    units of energy.
    """
    instrument_optika = _instrument_optika()
    images_optika = _images_optika(instrument_optika)

    guess = None
    if energy:
        guess = _guess_optika(instrument_optika, unit=unit_radiance)

    kwargs = dict(
        instrument=instrument_optika,
        num_iteration=6,
        threshold_convergence=-np.inf,
    )

    single = ctis.inverters.MartInverter(dtype=np.float32, **kwargs)
    double = ctis.inverters.MartInverter(**kwargs)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = single(images_optika, guess=guess)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = double(images_optika, guess=guess)

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


def test__call__optika_lines() -> None:
    """
    A scene of two spectral lines, each of which the sensor responds to in
    its own way, is reconstructed on the matrices of an instrument on a
    linear system of :mod:`optika`: the solution after a given number of
    iterations is that of the iteration on the methods of the instrument,
    it has the axis of the lines, and the merit is within the tolerance of
    its own.
    """
    instrument_lines = _instrument_optika_lines()
    a = ctis.inverters.MartInverter(
        instrument=instrument_lines,
        num_iteration=4,
        threshold_convergence=-np.inf,
    )
    images_lines = _images_optika(instrument_lines)

    assert _uses_operators(a, images_lines)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        result = a(images_lines)

    with pytest.warns(UserWarning, match="Max number of iterations"):
        expected = _reference(a, images_lines)

    assert result.solution.outputs.shape["line"] == 2
    assert result.num_iteration == expected.num_iteration == 4
    _assert_close(result.solution.outputs, expected.solution.outputs)

    ratio = result.mean_chi_squared / expected.mean_chi_squared
    error = np.abs(na.value(ratio).ndarray - 1)

    assert error.max() < instrument_lines._tolerance_vmr
    assert error.max() > 1e-8
