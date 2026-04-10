import pytest
import abc
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis

velocity = na.linspace(-500, 500, axis="wavelength", num=21) * u.km / u.s

position_scene = na.Cartesian2dVectorLinearSpace(
    start=-10 * u.arcsec,
    stop=10 * u.arcsec,
    axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
    num=na.Cartesian2dVectorArray(64, 64),
)

position_sensor = na.Cartesian2dVectorArray(
    x=na.arange(0, 64, axis="sensor_x") * u.pix,
    y=na.arange(0, 64, axis="sensor_y") * u.pix,
)

coordinates_scene = na.SpectralPositionalVectorArray(velocity, position_scene)
coordinates_sensor = na.SpectralPositionalVectorArray(velocity, position_sensor)

gaussians = ctis.scenes.gaussians(
    inputs=coordinates_scene,
    width=na.SpectralPositionalVectorArray(30 * u.km / u.s, 1 * u.arcsec),
)

wavelength_rest = 171 * u.AA

AA = dict(
    unit=u.AA,
    equivalencies=u.doppler_optical(wavelength_rest),
)

coordinates_scene.wavelength = coordinates_scene.wavelength.to(**AA)
coordinates_sensor.wavelength = coordinates_sensor.wavelength.to(**AA)

dispersion = 10 * u.km / u.s
dispersion = (dispersion.to(**AA) - wavelength_rest) / u.pix

instrument_ideal = ctis.instruments.IdealInstrument(
    area_effective=1 * u.cm**2,
    timedelta_exposure=10 * u.s,
    plate_scale=0.4 * u.arcsec / u.pix,
    dispersion=dispersion,
    angle=0 * u.deg,
    wavelength_ref=wavelength_rest,
    position_ref=32 * u.pix,
    coordinates_scene=coordinates_scene,
    coordinates_sensor=coordinates_sensor,
    axis_wavelength="wavelength",
    axis_scene_xy=("scene_x", "scene_y"),
    axis_sensor_xy=("sensor_x", "sensor_y"),
)


class AbstractTestAbstractInstrument(
    abc.ABC,
):

    @pytest.mark.parametrize(
        argnames="scene",
        argvalues=[
            gaussians.outputs,
        ],
    )
    def test_image(
        self,
        a: ctis.instruments.AbstractInstrument,
        scene: na.AbstractScalar,
    ):
        result = a.image(scene)
        assert np.all(result.inputs == coordinates_sensor)
        assert result.outputs.sum() > 0

    @pytest.mark.parametrize(
        argnames="image",
        argvalues=[
            instrument_ideal.image(gaussians.outputs).outputs,
        ],
    )
    def test_backproject(
        self,
        a: ctis.instruments.AbstractInstrument,
        image: na.AbstractScalar,
    ):
        result = a.backproject(image)
        assert np.all(result.inputs == coordinates_scene)
        assert result.outputs.sum() > 0


class AbstractTestAbstractLinearInstrument(
    AbstractTestAbstractInstrument,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[instrument_ideal],
)
class TestIdealInstrument(
    AbstractTestAbstractLinearInstrument,
):
    pass
