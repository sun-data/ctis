import pytest
import abc
import math
import dataclasses
import numpy as np
import scipy.ndimage
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
import ctis


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
        operators = a._operators
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
        assert a._operators is not None
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

        assert a._operators is not None
        super().test__operators(a)

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
        them, which leaves it to its `image` and `backproject` methods.
        """
        a = dataclasses.replace(a, **kwargs)

        assert a._operators is None

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

        assert a._operators is not None
        assert "weights" not in vars(a)
        assert "weights_transpose" not in vars(a)

        assert a.weights[0].shape == shape
        assert "weights" in vars(a)

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
        assert a._operators is None
        super().test__operators(a)

    def _with_read_noise(self, a, read_noise):
        sensor = dataclasses.replace(a.system.sensor, read_noise=read_noise)
        return dataclasses.replace(
            a,
            system=dataclasses.replace(a.system, sensor=sensor),
        )
