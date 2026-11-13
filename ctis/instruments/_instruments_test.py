import pytest
import abc
import math
import weakref
import dataclasses
import numpy as np
import scipy.ndimage
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
import ctis
from ctis.instruments._operators import Operators


def _scene(
    a: ctis.instruments.AbstractInstrument,
) -> na.FunctionArray[na.AbstractSpectralPositionalVectorArray, na.AbstractScalar]:
    """
    A compact spectral radiance defined on the scene grid of `a`.

    The signal occupies a single central field cell, so its dispersed image is
    contained within the sensor and the backprojection is contained within the
    scene grid. This lets the forward model conserve flux exactly through a
    round trip.
    """
    coordinates = a.coordinates_scene
    axis_x, axis_y = a.axis_scene_xy
    shape = na.shape(coordinates.position)

    values = np.zeros((shape[axis_x] - 1, shape[axis_y] - 1))
    values[values.shape[0] // 2, values.shape[1] // 2] = 1.0

    radiance = 1e-11 * u.erg / u.s / u.cm**2 / u.arcsec**2 / u.nm
    outputs = na.ScalarArray(values, axes=(axis_x, axis_y)) * radiance
    return na.FunctionArray(inputs=coordinates, outputs=outputs)


def _scene_frames(
    a: ctis.instruments.IdealInstrument,
    shape_frames: dict[str, int],
) -> na.FunctionArray[na.AbstractSpectralPositionalVectorArray, na.ScalarArray]:
    """
    A different random scene for each of several frames, defined on the scene
    grid of `a`, which fills the field of view.
    """
    operators = a._operators()
    values = na.random.uniform(
        low=1,
        high=2,
        shape_random=operators.shape_scene | shape_frames,
        seed=0,
    )
    return na.FunctionArray(
        inputs=a.coordinates_scene,
        outputs=values * operators.unit_scene,
    )


def _assert_close(result: na.AbstractScalar, expected: na.AbstractScalar) -> None:
    """
    Assert that two named arrays have the same axes, the same unit, and the
    same values to within the rounding of their arithmetic.
    """
    assert result.shape == expected.shape
    assert na.unit_normalized(result) == na.unit_normalized(expected)

    result = na.value(result).ndarray_aligned(expected.axes)
    expected = na.value(expected).ndarray

    assert np.allclose(
        result,
        expected,
        rtol=1e-10,
        atol=1e-10 * np.abs(expected).max(),
    )


def _assert_image_matrices(
    a: ctis.instruments.IdealInstrument,
    shape_frames: dict[str, int],
    function: bool = False,
) -> None:
    """
    Assert that the image integrated along wavelength, which `a` computes
    with its matrices, is the one it computes from its weights, with the same
    uncertainty, for the scene of every frame.

    Parameters
    ----------
    a
        The instrument to image the scenes with.
    shape_frames
        The axes of the frames and their lengths.
    function
        Whether to give the scenes with their coordinates, as a function,
        or as their values alone.
    """
    scene = _scene_frames(a, shape_frames)
    if not function:
        scene = scene.outputs

    kwargs = dict(noise=False, uncertainty=True)

    assert a._image_matrices(scene, **kwargs) is not None

    result = a.image(scene, **kwargs)
    expected = a._image_weights(scene, integrate=True, **kwargs)

    assert isinstance(result, na.FunctionArray)
    assert np.all(result.inputs == expected.inputs)
    assert isinstance(result.outputs, na.NormalUncertainScalarArray)
    assert result.outputs.nominal.shape == a._operators().shape_images | shape_frames
    assert result.outputs.nominal.sum() > 0
    _assert_close(result.outputs.nominal, expected.outputs.nominal)
    _assert_close(result.outputs.width, expected.outputs.width)

    result = a.image(scene, noise=False)

    assert isinstance(result.outputs, na.ScalarArray)
    _assert_close(result.outputs, expected.outputs.nominal)


def _assert_backproject_matrices(
    a: ctis.instruments.IdealInstrument,
    shape_frames: dict[str, int],
    function: bool = False,
) -> None:
    """
    Assert that the backprojection of images integrated along wavelength,
    which `a` computes with its matrices, is the one it computes from its
    weights, for the images of every frame.

    Parameters
    ----------
    a
        The instrument to backproject the images with.
    shape_frames
        The axes of the frames and their lengths.
    function
        Whether to give the images with their coordinates, as a function,
        or as their values alone.
    """
    image = a.image(_scene_frames(a, shape_frames), noise=False)
    if not function:
        image = image.outputs

    assert a._backproject_matrices(image) is not None

    result = a.backproject(image)
    expected = a._backproject_weights(image, integrate=True)

    operators = a._operators()
    shape = operators.shape_channel | operators.shape_scene | shape_frames

    assert isinstance(result, na.FunctionArray)
    assert np.all(result.inputs == expected.inputs)
    assert result.outputs.shape == shape
    assert result.outputs.sum() > 0
    _assert_close(result.outputs, expected.outputs)


class AbstractTestAbstractInstrument(
    abc.ABC,
):

    def test_image(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        scene = _scene(a)
        result = a.image(scene.outputs, noise=False)
        assert isinstance(result, na.FunctionArray)
        assert np.all(result.inputs.position == a.coordinates_sensor.position)
        assert result.outputs.sum() > 0

    def test_backproject(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        scene = _scene(a)
        image = a.image(scene.outputs, noise=False)
        result = a.backproject(image)

        assert isinstance(result, na.FunctionArray)
        assert np.all(result.inputs == a.coordinates_scene)
        assert np.all(np.isfinite(na.as_named_array(result.outputs).value))
        assert result.outputs.sum() > 0

    def test_backproject_scalar(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        # backproject also accepts the bare image outputs, not just a
        # `FunctionArray`
        scene = _scene(a)
        image = a.image(scene.outputs, noise=False)
        result = a.backproject(image.outputs)

        assert isinstance(result, na.FunctionArray)
        assert np.all(np.isfinite(na.as_named_array(result.outputs).value))

    def test_backproject_unit(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        scene = _scene(a)
        image = a.image(scene.outputs, noise=False)

        # the backprojection can be expressed in the (energy) units of the
        # input scene, or in photon units, to match the forward model.
        unit_energy = na.unit_normalized(scene.outputs)
        unit_photon = u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm

        result_energy = a.backproject(image, unit=unit_energy)
        result_photon = a.backproject(image, unit=unit_photon)

        assert na.unit_normalized(result_energy.outputs).is_equivalent(unit_energy)
        assert na.unit_normalized(result_photon.outputs).is_equivalent(unit_photon)

        # the two expressions differ only by the energy per photon
        wavelength = a.coordinates_scene.wavelength.cell_centers(a.axis_wavelength)
        energy_per_photon = (
            astropy.constants.h * astropy.constants.c / wavelength / u.photon
        )
        expected_energy = (result_photon.outputs * energy_per_photon).to(unit_energy)
        assert np.allclose(result_energy.outputs, expected_energy)

        # a unit compatible with neither photon nor energy radiance is rejected
        with pytest.raises(u.UnitConversionError):
            a.backproject(image, unit=u.s)

    def test_backproject_conserves_flux(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        # backproject is the adjoint of the forward model, so for a scene
        # contained within the field of view, re-imaging a backprojection
        # preserves the total number of measured electrons.
        scene = _scene(a)
        image = a.image(scene.outputs, noise=False)
        result = a.backproject(image)
        image_check = a.image(result, noise=False)
        assert np.allclose(
            image.outputs.sum().ndarray.value,
            image_check.outputs.sum().ndarray.value,
        )

    def test_num_channel(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        result = a.num_channel

        assert isinstance(result, int)

    def test_axis_sensor_xy(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        result = a.axis_sensor_xy

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(ax, str) for ax in result)

    @pytest.mark.parametrize("integrate", [False, True])
    def test_image_uncertainty(
        self,
        a: ctis.instruments.AbstractInstrument,
        integrate: bool,
    ):
        scene = _scene(a)
        result = a.image(
            scene.outputs,
            integrate=integrate,
            noise=False,
            uncertainty=True,
        )

        # the measurement noise is attached as a normal uncertain array
        assert isinstance(result.outputs, na.NormalUncertainScalarArray)
        assert result.outputs.width.unit.is_equivalent(u.electron)
        assert np.all(result.outputs.width >= 0 * u.electron)

        # the width is deterministic: it is computed from the expected signal,
        # so drawing a noise realization must not change it.
        result_noisy = a.image(
            scene.outputs,
            integrate=integrate,
            noise=True,
            uncertainty=True,
        )
        assert np.allclose(
            result_noisy.outputs.width.to_value(u.electron),
            result.outputs.width.to_value(u.electron),
        )

    def test__operators(
        self,
        a: ctis.instruments.AbstractInstrument,
    ) -> None:
        # the operators of an instrument, if it provides them, reproduce its
        # `image` and `backproject` methods on plain arrays
        operators = a._operators()
        if operators is None:
            return

        shape_scene = operators.shape_scene
        shape_images = operators.shape_images
        shape_voxels = operators.shape_channel | shape_scene

        assert operators.num_channel == a.num_channel

        rng = np.random.default_rng(0)
        scene = rng.uniform(1, 2, size=tuple(shape_scene.values()))

        expected = a.image(
            scene=na.ScalarArray(
                ndarray=scene << operators.unit_scene,
                axes=tuple(shape_scene),
            ),
            noise=False,
            uncertainty=True,
        ).outputs
        nominal = expected.nominal.ndarray_aligned(tuple(shape_images))
        nominal = nominal.to_value(operators.unit_images)
        variance_expected = np.square(expected.width)
        variance_expected = variance_expected.ndarray_aligned(tuple(shape_images))
        variance_expected = variance_expected.to_value(operators.unit_images**2)

        # the operators take the frames along a last axis
        images, variance = operators.image(scene[..., np.newaxis])

        assert images.shape == tuple(shape_images.values()) + (1,)
        assert variance.shape == tuple(shape_images.values()) + (1,)

        images = images[..., 0]
        variance = variance[..., 0]

        assert images.sum() > 0
        assert np.allclose(images, nominal, rtol=1e-10, atol=1e-10 * nominal.max())
        assert np.allclose(
            variance,
            variance_expected,
            rtol=1e-10,
            atol=1e-10 * variance_expected.max(),
        )

        expected = a.backproject(
            image=na.ScalarArray(
                ndarray=images << operators.unit_images,
                axes=tuple(shape_images),
            ),
        ).outputs
        expected = expected.ndarray_aligned(tuple(shape_voxels))
        expected = expected.to_value(operators.unit_scene)

        result = operators.backproject(images[..., np.newaxis])

        assert result.shape == tuple(shape_voxels.values()) + (1,)
        assert result.sum() > 0
        assert np.allclose(
            (result * operators.scale_backproject)[..., 0],
            expected,
            rtol=1e-10,
            atol=1e-10 * expected.max(),
        )

        # each frame of several gets the result it would have alone
        num_frame = 3
        scenes = rng.uniform(1, 2, size=tuple(shape_scene.values()) + (num_frame,))
        images, variance = operators.image(scenes)
        result = operators.backproject(images)
        for f in range(num_frame):
            images_f, variance_f = operators.image(scenes[..., f : f + 1])
            result_f = operators.backproject(images_f)
            assert np.array_equal(images[..., f : f + 1], images_f)
            assert np.array_equal(variance[..., f : f + 1], variance_f)
            assert np.array_equal(result[..., f : f + 1], result_f)

    @abc.abstractmethod
    def _with_read_noise(
        self,
        a: ctis.instruments.AbstractInstrument,
        read_noise: u.Quantity,
    ) -> ctis.instruments.AbstractInstrument:
        """Return a copy of `a` with the given per-readout read noise."""

    def test_read_noise(
        self,
        a: ctis.instruments.AbstractInstrument,
    ):
        # read noise is added once per readout, so it raises the integrated
        # uncertainty by exactly `read_noise` in quadrature (not by
        # sqrt(num_wavelength) * read_noise)
        scene = _scene(a)
        a_rn = self._with_read_noise(a, 10 * u.electron)
        width_0 = a.image(scene.outputs, noise=False, uncertainty=True).outputs.width
        width_1 = a_rn.image(scene.outputs, noise=False, uncertainty=True).outputs.width
        contribution = np.sqrt(np.square(width_1) - np.square(width_0))
        assert np.allclose(contribution.to_value(u.electron), 10)


def _psf(a: ctis.instruments.IdealInstrument) -> na.ScalarArray:
    """
    A different, asymmetric point-spread function for each channel of `a`,
    each normalized to a total of one.
    """
    axis_x, axis_y = a.axis_psf_xy
    result = na.random.uniform(
        low=0,
        high=1,
        shape_random={a.axis_channel: a.num_channel, axis_x: 3, axis_y: 5},
        seed=0,
    )
    return result / result.sum((axis_x, axis_y))


class AbstractTestAbstractLinearInstrument(
    AbstractTestAbstractInstrument,
):
    def test_num_channel(
        self,
        a: ctis.instruments.AbstractLinearInstrument,
    ) -> None:
        super().test_num_channel(a)

        # the channels are those that the weights are an array over
        axis_channel = a.axis_channel
        if isinstance(axis_channel, str):
            axis_channel = (axis_channel,)
        shape = a.weights[0].shape

        assert a.num_channel == math.prod(shape[ax] for ax in axis_channel)


velocity = na.linspace(-500, 500, axis="wavelength", num=21) * u.km / u.s

wavelength_rest = 171 * u.AA

position_scene = na.Cartesian2dVectorLinearSpace(
    start=-20 * u.arcsec,
    stop=20 * u.arcsec,
    axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
    num=na.Cartesian2dVectorArray(64, 64),
)

position_sensor = na.Cartesian2dVectorArray(
    x=na.arange(0, 64, axis="sensor_x") * u.pix,
    y=na.arange(0, 64, axis="sensor_y") * u.pix,
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

AA = dict(
    unit=u.AA,
    equivalencies=u.doppler_optical(wavelength_rest),
)

coordinates_scene.wavelength = coordinates_scene.wavelength.to(**AA)
coordinates_sensor.wavelength = coordinates_sensor.wavelength.to(**AA)

dispersion = 200 * u.km / u.s
dispersion = (dispersion.to(**AA) - wavelength_rest) / u.pix

angle = na.linspace(0, 360, axis="channel", num=3, endpoint=False)

instrument_ideal = ctis.instruments.IdealInstrument(
    area_effective=1 * u.cm**2,
    timedelta_exposure=10 * u.s,
    plate_scale=2 * u.arcsec / u.pix,
    dispersion=dispersion,
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


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        instrument_ideal,
        dataclasses.replace(instrument_ideal, psf=_psf(instrument_ideal)),
    ],
    ids=["no psf", "psf"],
)
class TestIdealInstrument(
    AbstractTestAbstractLinearInstrument,
):
    def _with_read_noise(self, a, read_noise):
        return dataclasses.replace(a, read_noise=read_noise)

    def test__operators(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        assert a._operators() is not None
        super().test__operators(a)

    @pytest.mark.parametrize(
        argnames="kwargs",
        argvalues=[
            dict(
                area_effective=na.linspace(1, 2, axis="channel", num=3) * u.cm**2,
                timedelta_exposure=na.linspace(10, 20, axis="channel", num=3) * u.s,
            ),
            dict(
                quantum_yield=na.linspace(2, 3, axis="channel", num=3)
                * u.electron
                / u.photon,
            ),
            dict(
                read_noise=na.random.uniform(
                    low=1 * u.electron,
                    high=2 * u.electron,
                    shape_random=dict(channel=3, sensor_x=63, sensor_y=63),
                    seed=0,
                ),
            ),
            dict(quantum_yield=2, read_noise=3 * u.photon),
            dict(quantum_yield=2 * u.DN / u.photon, read_noise=3 * u.DN),
        ],
        ids=[
            "exposure for each channel",
            "quantum yield for each channel",
            "read noise for each pixel",
            "images in photons",
            "images in DN",
        ],
    )
    def test__operators_varying(
        self,
        a: ctis.instruments.IdealInstrument,
        kwargs: dict,
    ) -> None:
        """
        The operators reproduce an instrument whose exposure, quantum yield
        or read noise vary along the axes that the operators can express,
        and one whose images are not in electrons.
        """
        a = dataclasses.replace(a, **kwargs)

        assert a._operators() is not None
        super().test__operators(a)

        # and its `image` and `backproject` methods, which are computed with
        # the matrices, are those computed from its weights
        _assert_image_matrices(a, shape_frames=dict(time=2))
        _assert_backproject_matrices(a, shape_frames=dict(time=2))

    @pytest.mark.parametrize(
        argnames="kwargs",
        argvalues=[
            dict(
                quantum_yield=na.ScalarArray(np.ones(20), axes="wavelength")
                * u.electron
                / u.photon,
            ),
            dict(
                plate_scale=na.linspace(2, 2.1, axis="time", num=2) * u.arcsec / u.pix,
            ),
            dict(read_noise=0),
        ],
        ids=[
            "quantum yield for each wavelength",
            "plate scale for each time",
            "read noise without a unit",
        ],
    )
    def test__operators_unsupported(
        self,
        a: ctis.instruments.IdealInstrument,
        kwargs: dict,
    ) -> None:
        """
        An instrument which the operators cannot express does not provide
        them, which leaves its images and backprojections to its weights.
        """
        a = dataclasses.replace(a, **kwargs)

        assert a._operators() is None

        scene = _scene(a)

        assert a._image_matrices(scene, noise=False, uncertainty=False) is None

        image = a.image(scene, noise=False)

        assert image.outputs.sum() > 0
        assert a._backproject_matrices(image) is None
        assert a.backproject(image).outputs.sum() > 0

    def test__operators_weights(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        Assembling the operators releases the weights, which hold the same
        matrices in another form, and which are computed again if needed.
        """
        a = dataclasses.replace(a)

        shape = a.weights[0].shape
        a.weights_transpose
        assert "weights" in vars(a)
        assert "weights_transpose" in vars(a)

        assert a._operators() is not None
        assert "weights" not in vars(a)
        assert "weights_transpose" not in vars(a)

        assert a.weights[0].shape == shape
        assert "weights" in vars(a)

    def test__operators_backward(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        The matrix of the backprojection is derived the first time it is
        needed, which an image does not.
        """
        a = dataclasses.replace(a)

        scene = _scene(a)
        image = a.image(scene, noise=False)
        operators = a._operators()

        assert operators._backward is None
        assert "weights" not in vars(a)

        a.backproject(image)
        backward = operators._backward

        assert backward is not None
        assert operators.backward is backward
        assert "weights" not in vars(a)
        assert "weights_transpose" not in vars(a)

    def test__operators_photons(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        The photons leaving every voxel are written into the buffer that a
        call is given, which gives the photons that a call without one does,
        and the operators keep no array of their own between calls.
        """
        a = dataclasses.replace(a)
        operators = a._operators()

        rng = np.random.default_rng(0)
        scene = rng.uniform(1, 2, size=tuple(operators.shape_scene.values()) + (2,))

        photons = operators.photons(scene)

        buffer = np.zeros(operators.forward.shape_input + (2,))
        photons_buffer = operators.photons(scene, buffer=buffer)

        assert np.array_equal(photons_buffer, photons)
        assert np.all(buffer.reshape(operators._scale.shape + (2,)) > 0)

        images, variance = operators.image(scene, buffer=buffer)

        assert np.array_equal(images, photons * operators.gain)
        assert not any(
            isinstance(value, np.ndarray) and value.size == buffer.size
            for value in vars(operators).values()
        )

        # a buffer of another shape, of another precision or which is not
        # contiguous would be read or written out of step
        for wrong in (
            np.zeros(operators.forward.shape_input + (1,)),
            buffer.astype(np.float32),
            np.zeros(operators.forward.shape_input + (4,))[..., ::2],
        ):
            with pytest.raises(ValueError, match="the buffer should be"):
                operators.photons(scene, buffer=wrong)

    def test__operators_shape(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        The operators refuse scenes and images of the wrong shape, which
        their compiled code would otherwise read out of bounds.
        """
        operators = a._operators()

        shape_scene = tuple(operators.shape_scene.values())
        shape_images = tuple(operators.shape_images.values())

        with pytest.raises(ValueError, match="the scenes should have a shape of"):
            operators.photons(np.ones(shape_scene[1:] + (1,)))

        with pytest.raises(ValueError, match="the scenes should have a shape of"):
            operators.image(np.ones(shape_scene))

        with pytest.raises(ValueError, match="cannot reshape array"):
            operators.backproject(np.ones(shape_images[1:] + (1,)))

    @pytest.mark.parametrize(
        argnames="shape_frames",
        argvalues=[dict(), dict(time=2), dict(time=2, repeat=3)],
        ids=["one frame", "an axis of frames", "two axes of frames"],
    )
    @pytest.mark.parametrize(
        argnames="function",
        argvalues=[False, True],
        ids=["values", "function"],
    )
    def test_image_matrices(
        self,
        a: ctis.instruments.IdealInstrument,
        shape_frames: dict[str, int],
        function: bool,
    ) -> None:
        """
        The image integrated along wavelength is computed with the matrices,
        and is the one computed from the weights.
        """
        _assert_image_matrices(a, shape_frames=shape_frames, function=function)

    @pytest.mark.parametrize(
        argnames="shape_frames",
        argvalues=[dict(), dict(time=2), dict(time=2, repeat=3)],
        ids=["one frame", "an axis of frames", "two axes of frames"],
    )
    @pytest.mark.parametrize(
        argnames="function",
        argvalues=[False, True],
        ids=["values", "function"],
    )
    def test_backproject_matrices(
        self,
        a: ctis.instruments.IdealInstrument,
        shape_frames: dict[str, int],
        function: bool,
    ) -> None:
        """
        The backprojection of images integrated along wavelength is computed
        with the matrices, and is the one computed from the weights.
        """
        _assert_backproject_matrices(a, shape_frames=shape_frames, function=function)

    @pytest.mark.parametrize(
        argnames="matrices",
        argvalues=[True, False],
        ids=["matrices", "weights"],
    )
    def test_image_noise(
        self,
        a: ctis.instruments.IdealInstrument,
        matrices: bool,
    ) -> None:
        """
        The noise of an image has the mean and the variance that its
        uncertainty says, whether it is drawn once for each pixel, as it is
        with the matrices, or for each wavelength, as it is from the weights,
        and it is drawn independently for each frame.
        """
        quantum_yield = 3 * u.electron / u.photon
        a = dataclasses.replace(
            a,
            quantum_yield=quantum_yield,
            read_noise=5 * u.electron,
        )

        # the same scene in every frame, dim enough for its shot noise to
        # be skewed, and bright enough to outweigh the read noise
        scene = _scene_frames(a, shape_frames=dict()).outputs
        brightest = a.image(scene, noise=False).outputs.max() / quantum_yield
        scene = scene * (30 / brightest.ndarray.to_value(u.photon))
        scene = na.broadcast_to(scene, scene.shape | dict(repeat=20))

        kwargs = dict(noise=True, uncertainty=True)

        state = np.random.get_state()
        np.random.seed(0)
        try:
            if matrices:
                result = a.image(scene, **kwargs).outputs
            else:
                result = a._image_weights(scene, integrate=True, **kwargs).outputs
        finally:
            np.random.set_state(state)

        expected = a.image(scene, noise=False, uncertainty=True).outputs

        _assert_close(result.width, expected.width)

        residual = (result.nominal - expected.nominal) / expected.width
        residual = residual.ndarray
        num = residual.size

        # the frames are separate realizations of the noise
        axes = tuple(ax for ax in result.nominal.axes if ax != "repeat")
        frames = na.value(result.nominal).ndarray_aligned(("repeat",) + axes)
        assert not np.any(np.all(frames[0] == frames[1:], axis=(1, 2, 3)))

        # the mean and the variance of the residuals are within five standard
        # errors of zero and of one, the kurtosis of the residuals being
        # under four
        assert np.abs(residual.mean()) < 5 / np.sqrt(num)
        assert np.abs(residual.var() - 1) < 5 * np.sqrt(3 / num)

    @pytest.mark.parametrize(
        argnames="matrices",
        argvalues=[True, False],
        ids=["matrices", "weights"],
    )
    def test_image_noise_photons(
        self,
        a: ctis.instruments.IdealInstrument,
        matrices: bool,
    ) -> None:
        """
        Without read noise, the image is a whole number of photons in every
        pixel, each converted into the electrons of the quantum yield,
        whether it is computed with the matrices or from the weights.
        """
        quantum_yield = 3 * u.electron / u.photon
        a = dataclasses.replace(a, quantum_yield=quantum_yield)

        scene = _scene_frames(a, shape_frames=dict()).outputs
        brightest = a.image(scene, noise=False).outputs.max() / quantum_yield
        scene = scene * (30 / brightest.ndarray.to_value(u.photon))

        if matrices:
            result = a.image(scene).outputs
        else:
            result = a._image_weights(
                scene=scene,
                integrate=True,
                noise=True,
                uncertainty=False,
            ).outputs
        photons = (result / quantum_yield).ndarray.to_value(u.photon)

        assert result.unit == u.electron
        assert photons.sum() > 0
        assert np.array_equal(photons, np.round(photons))

    def test_image_matrices_unsupported(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        A scene which the matrices cannot take is left to the weights, which
        image it, or raise the error they always have.
        """
        operators = a._operators()
        kwargs = dict(noise=False, uncertainty=False)

        # a scene for each channel
        scene = _scene_frames(a, shape_frames=operators.shape_channel)

        assert a._image_matrices(scene, **kwargs) is None
        assert a.image(scene, noise=False).outputs.sum() > 0

        # a scene which is not in units of energy
        scene = _scene(a).outputs / u.erg

        assert a._image_matrices(scene, **kwargs) is None
        with pytest.raises(u.UnitConversionError):
            a.image(scene, noise=False)

        # a scene on another grid
        scene = _scene(a)
        scene = scene.replace(
            inputs=scene.inputs.replace(position=2 * scene.inputs.position)
        )

        with pytest.raises(ValueError, match="coordinates_scene"):
            a.image(scene, noise=False)

    def test_backproject_matrices_unsupported(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        An image which the matrices cannot take is left to the weights, which
        backproject it, or raise the error they always have.
        """
        scene = _scene(a)

        # an image of each wavelength
        image = a.image(scene, integrate=False, noise=False)

        assert a._backproject_matrices(image.outputs) is None
        assert a.backproject(image.outputs, integrate=False).outputs.sum() > 0

        # an image without a unit
        image = a.image(scene, noise=False)

        assert a._backproject_matrices(na.value(image.outputs)) is None
        with pytest.raises(u.UnitConversionError):
            a.backproject(na.value(image.outputs))

        # an image on another grid
        image = image.replace(
            inputs=image.inputs.replace(position=2 * image.inputs.position)
        )

        with pytest.raises(ValueError, match="coordinates_sensor"):
            a.backproject(image)

    def test_image_separate(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        With the wavelengths kept separate, the images and the
        backprojections are computed from the weights, and the matrices are
        not assembled.
        """
        a = dataclasses.replace(a)
        scene = _scene(a)

        image = a.image(scene, integrate=False, noise=False)
        result = a.backproject(image, integrate=False)

        assert a.axis_wavelength in image.outputs.shape
        assert result.outputs.sum() > 0
        assert a._operators_kept is None
        assert "weights" in vars(a)
        assert "weights_transpose" in vars(a)

    def test__operators_single(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        The operators are assembled in the precision asked for, and kept in
        the precision they were last asked for in.
        """
        a = dataclasses.replace(a)

        double = a._operators()
        assert double.dtype == np.float64
        assert a._operators() is double

        single = a._operators(np.float32)
        assert single.dtype == np.float32
        assert a._operators(np.float32) is single
        for array in (
            single.forward.data,
            single.backward.data,
            single.scale,
            single.scale_backproject,
            single.gain,
            single.variance_read,
        ):
            assert array.dtype == np.float32

        rng = np.random.default_rng(0)
        scene = rng.uniform(1, 2, size=tuple(single.shape_scene.values()) + (2,))

        images, variance = single.image(scene.astype(np.float32))
        images_expected, variance_expected = double.image(scene)
        result = single.backproject(images)
        result_expected = double.backproject(images_expected)

        for array, expected in (
            (images, images_expected),
            (variance, variance_expected),
            (result, result_expected),
        ):
            assert array.dtype == np.float32
            assert np.allclose(array, expected, rtol=1e-5, atol=1e-5 * expected.max())

        # the operators in double precision were released for those in single
        assert a._operators() is not double

    def test__operators_released(
        self,
        a: ctis.instruments.IdealInstrument,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The operators kept in one precision are released before those of
        another precision are assembled, so that the two never have to fit
        in memory together.
        """
        a = dataclasses.replace(a)

        kept = weakref.ref(a._operators())
        assert kept() is not None

        released = []
        operators_new = a._operators_new

        def spy(dtype: np.dtype) -> None | Operators:
            released.append(kept() is None)
            return operators_new(dtype)

        monkeypatch.setattr(a, "_operators_new", spy)

        assert a._operators(np.float32).dtype == np.float32
        assert released == [True]

    @pytest.mark.parametrize(
        argnames="name,value",
        argvalues=[
            ("area_effective", 3 * u.cm**2),
            ("timedelta_exposure", 40 * u.s),
            ("quantum_yield", 2.5 * u.electron / u.photon),
            ("read_noise", 4 * u.electron),
        ],
        ids=["effective area", "exposure time", "quantum yield", "read noise"],
    )
    def test__operators_assigned(
        self,
        a: ctis.instruments.IdealInstrument,
        name: str,
        value: u.Quantity,
    ) -> None:
        """
        A field which is assigned after the operators were assembled takes
        effect: the factors around the matrices are computed again, on the
        same matrices, and the images and the backprojections are those of
        an instrument which had the value from the start.
        """
        a = dataclasses.replace(a)
        scene = _scene_frames(a, shape_frames=dict())

        operators = a._operators()
        a.backproject(a.image(scene, noise=False))
        backward = operators._backward

        assert backward is not None
        assert a._operators() is operators

        setattr(a, name, value)

        operators_new = a._operators()

        assert operators_new is not operators
        assert operators_new.forward is operators.forward
        assert operators_new._backward is backward
        assert a._operators() is operators_new
        assert "weights" not in vars(a)

        # the weights of an instrument which never had another value
        expected = dataclasses.replace(a)
        kwargs = dict(noise=False, uncertainty=True)

        image = a.image(scene, **kwargs).outputs
        image_expected = expected._image_weights(scene, integrate=True, **kwargs)
        image_expected = image_expected.outputs

        assert a._operators() is operators_new
        _assert_close(image.nominal, image_expected.nominal)
        _assert_close(image.width, image_expected.width)

        _assert_close(
            a.backproject(image.nominal).outputs,
            expected._backproject_weights(image.nominal, integrate=True).outputs,
        )

    def test__operators_assigned_unsupported(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        An instrument stops providing its operators once a field is assigned
        a value which they cannot express, and provides them again once it
        is assigned one which they can.
        """
        a = dataclasses.replace(a)
        quantum_yield = a.quantum_yield

        assert a._operators() is not None

        a.quantum_yield = (
            na.ScalarArray(np.ones(20), axes="wavelength") * u.electron / u.photon
        )

        assert a._operators() is None
        assert a._operators() is None

        a.quantum_yield = quantum_yield

        assert a._operators() is not None

    def test__operators_subclass(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        A subclass which replaces a method that the operators stand in for
        does not provide them, so that it is imaged and inverted with its own
        methods, and a subclass which replaces none of them still does.
        """

        @dataclasses.dataclass
        class Dimmed(ctis.instruments.IdealInstrument):
            """An ideal instrument which measures half of the signal."""

            def image(
                self,
                scene: na.AbstractScalar | na.AbstractFunctionArray,
                integrate: bool = True,
                noise: bool = True,
                uncertainty: bool = False,
            ) -> na.FunctionArray:
                result = super().image(
                    scene=scene,
                    integrate=integrate,
                    noise=noise,
                    uncertainty=uncertainty,
                )
                return result.replace(outputs=result.outputs / 2)

        @dataclasses.dataclass
        class Renamed(ctis.instruments.IdealInstrument):
            """An ideal instrument by another name."""

        kwargs = {
            field.name: getattr(a, field.name)
            for field in dataclasses.fields(a)
            if field.init
        }
        scene = _scene(a)
        image = a.image(scene, noise=False).outputs

        dimmed = Dimmed(**kwargs)

        assert Dimmed._overridden()
        assert dimmed._operators() is None
        assert dimmed._operators(np.float32) is None
        _assert_close(dimmed.image(scene, noise=False).outputs, image / 2)

        renamed = Renamed(**kwargs)

        assert not Renamed._overridden()
        assert renamed._operators() is not None
        _assert_close(renamed.image(scene, noise=False).outputs, image)

    def test_num_channel_psf_sensor(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        The channels are counted, without computing the weights, on an
        instrument whose point-spread function varies across the sensor,
        along which the grids hold vertices where it holds cells.
        """
        axis_x, axis_y = a.axis_psf_xy
        shape_sensor = a._operators().shape_sensor

        psf = na.random.uniform(
            low=0,
            high=1,
            shape_random=shape_sensor | {axis_x: 3, axis_y: 3},
            seed=0,
        )
        a = dataclasses.replace(a, psf=psf / psf.sum((axis_x, axis_y)))

        num_channel = a.num_channel

        assert "weights" not in vars(a)
        assert num_channel == a.weights[0].shape[a.axis_channel]
        assert ctis.inverters.MartInverter(instrument=a).gamma == 2 / num_channel

    def test_image_negative(
        self,
        a: ctis.instruments.IdealInstrument,
    ) -> None:
        """
        The photons of a pixel are kept from being negative once they are
        summed over wavelength, with the matrices and without them, so a
        scene which is negative at some wavelengths has the same image and
        the same uncertainty either way, which are not those of each
        wavelength kept from being negative on its own.
        """
        axis = a.axis_wavelength
        num_wavelength = a._operators().shape_scene[axis]

        sign = np.where(np.arange(num_wavelength) < num_wavelength // 2, -3.0, 1.0)
        scene = _scene(a).outputs * na.ScalarArray(sign, axes=axis)

        kwargs = dict(noise=False, uncertainty=True)

        assert a._image_matrices(scene, **kwargs) is not None

        result = a.image(scene, **kwargs).outputs
        expected = a._image_weights(scene, integrate=True, **kwargs).outputs

        assert np.all(result.nominal >= 0)
        assert result.nominal.sum() > 0
        _assert_close(result.nominal, expected.nominal)
        _assert_close(result.width, expected.width)

        # without the sum, every wavelength is an image of its own
        separate = a.image(scene, integrate=False, noise=False).outputs

        assert np.all(separate >= 0)

        difference = np.abs(separate.sum(axis) - expected.nominal)

        assert difference.max() > 0.1 * expected.nominal.max()

        # with noise, each wavelength is drawn on its own, from an
        # expectation which is not negative, and the width is unchanged
        noisy = a._image_weights(scene, integrate=True, noise=True, uncertainty=True)

        assert np.all(np.isfinite(na.value(noisy.outputs.nominal)))
        assert np.all(noisy.outputs.nominal >= 0)
        _assert_close(noisy.outputs.width, expected.width)

    def test_psf(
        self,
        a: ctis.instruments.IdealInstrument,
    ):
        """
        The image of each wavelength with a point-spread function is the image
        without one, convolved with the point-spread function of its channel.
        """
        psf = _psf(a)
        a_psf = dataclasses.replace(a, psf=psf)
        a_none = dataclasses.replace(a, psf=None)

        scene = _scene(a)
        axes = (a.axis_channel, a.axis_wavelength, *a.axis_sensor_xy)
        actual = a_psf.image(scene.outputs, integrate=False, noise=False).outputs
        image = a_none.image(scene.outputs, integrate=False, noise=False).outputs

        actual = na.value(actual).ndarray_aligned(axes)
        image = na.value(image).ndarray_aligned(axes)
        kernel = psf.ndarray_aligned((a.axis_channel, *a.axis_psf_xy))
        expected = np.stack(
            [
                scipy.ndimage.convolve(image[c], kernel[c][np.newaxis], mode="constant")
                for c in range(a.num_channel)
            ]
        )

        assert actual.sum() > 0
        assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10 * expected.max())


def test_num_channel_axes_shared() -> None:
    """
    The channels are counted, without computing the weights, on an
    instrument whose scene and sensor have axes of the same names, along
    which their grids have different lengths.
    """
    axis_xy = ("x", "y")

    scene = na.DopplerPositionalVectorArray.from_velocity(
        velocity=velocity,
        wavelength_rest=wavelength_rest,
        position=na.Cartesian2dVectorLinearSpace(
            start=-20 * u.arcsec,
            stop=20 * u.arcsec,
            axis=na.Cartesian2dVectorArray(*axis_xy),
            num=na.Cartesian2dVectorArray(24, 24),
        ),
    )
    sensor = na.DopplerPositionalVectorArray.from_velocity(
        velocity=velocity,
        wavelength_rest=wavelength_rest,
        position=na.Cartesian2dVectorArray(
            x=na.arange(0, 49, axis="x") * u.pix,
            y=na.arange(0, 33, axis="y") * u.pix,
        ),
    )
    scene.wavelength = scene.wavelength.to(**AA)
    sensor.wavelength = sensor.wavelength.to(**AA)

    a = dataclasses.replace(
        instrument_ideal,
        coordinates_scene=scene,
        coordinates_sensor=sensor,
        axis_scene_xy=axis_xy,
        axis_sensor_xy=axis_xy,
    )

    num_channel = a.num_channel

    assert "weights" not in vars(a)
    assert num_channel == a.weights[0].shape[a.axis_channel]


def _instrument_optika() -> ctis.instruments.OptikaInstrument:
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
    # the scene grid spans the field of view so the backprojection of a
    # contained source is not clipped
    coordinates_scene = na.SpectralPositionalVectorArray(
        wavelength=na.linspace(530, 570, axis="wavelength", num=4) * u.nm,
        position=na.Cartesian2dVectorLinearSpace(
            start=-20 * u.arcsec,
            stop=+20 * u.arcsec,
            axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
            num=21,
        ),
    )
    return ctis.instruments.OptikaInstrument(
        system=system,
        coordinates_scene=coordinates_scene,
        channel=channel,
        axis_channel="channel",
        axis_wavelength="wavelength",
        axis_scene_xy=("scene_x", "scene_y"),
    )


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[_instrument_optika()],
)
class TestOptikaInstrument(
    AbstractTestAbstractLinearInstrument,
):
    def test__operators(
        self,
        a: ctis.instruments.OptikaInstrument,
    ) -> None:
        # the operators of this instrument are not implemented yet
        assert a._operators() is None
        super().test__operators(a)

    def _with_read_noise(self, a, read_noise):
        sensor = dataclasses.replace(a.system.sensor, read_noise=read_noise)
        return dataclasses.replace(
            a,
            system=dataclasses.replace(a.system, sensor=sensor),
        )
