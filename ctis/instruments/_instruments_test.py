import pytest
import abc
import numpy as np
import astropy.units as u
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


class AbstractTestAbstractLinearInstrument(
    AbstractTestAbstractInstrument,
):
    pass


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
    argvalues=[instrument_ideal],
)
class TestIdealInstrument(
    AbstractTestAbstractLinearInstrument,
):
    pass


def _instrument_optika() -> ctis.instruments.OptikaInstrument:
    channel = na.linspace(0, 360, axis="channel", num=3, endpoint=False) * u.deg
    system = optika.systems.LinearSystem(
        area_effective=optika.radiometry.InterpolatedEffectiveAreaModel(
            wavelength=na.linspace(400, 700, axis="wavelength", num=10) * u.nm,
            area=na.linspace(1, 2, axis="wavelength", num=10) * u.cm**2,
            axis_wavelength="wavelength",
        ),
        distortion=optika.distortion.SimpleDistortionModel(
            plate_scale=50 * u.arcsec / u.mm,
            dispersion=250 * u.nm / u.mm,
            angle=channel,
            reference=na.SpectralPositionalVectorArray(
                wavelength=550 * u.nm,
                position=na.Cartesian2dVectorArray(0, 0) * u.mm,
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
    pass
