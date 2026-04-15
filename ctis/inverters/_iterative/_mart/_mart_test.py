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

coordinates_scene = na.SpectralPositionalVectorArray(velocity, position_scene)
coordinates_sensor = na.SpectralPositionalVectorArray(velocity, position_sensor)

scene = ctis.scenes.gaussians(
    inputs=coordinates_scene,
    width=na.SpectralPositionalVectorArray(30 * u.km / u.s, 1 * u.arcsec),
)

coordinates_scene.wavelength = wavelength
coordinates_sensor.wavelength = wavelength

instrument = ctis.instruments.IdealInstrument(
    area_effective=1 * u.cm**2,
    timedelta_exposure=20 * u.s,
    plate_scale=0.4 * u.arcsec / u.pix,
    dispersion=((10 * u.km / u.s).to(**AA) - wavelength_rest) / u.pix,
    angle=na.linspace(0, 360, num=4, axis="channel", endpoint=False) * u.deg,
    wavelength_ref=wavelength_rest,
    position_ref=na.Cartesian2dVectorArray(64, 32) * u.pix,
    coordinates_scene=coordinates_scene,
    coordinates_sensor=coordinates_sensor,
    axis_channel="channel",
    axis_wavelength="wavelength",
    axis_scene_xy=("scene_x", "scene_y"),
    axis_sensor_xy=("sensor_x", "sensor_y"),
)

images = instrument.image(scene.outputs)

inverter = ctis.inverters.MartInverter(
    instrument=instrument,
)

@pytest.mark.parametrize("a", [inverter])
class TestMartInverter(
    AbstractTestAbstractIterativeInverter,
):

    @pytest.mark.parametrize("images", [images])
    def test__call__(
        self,
        a: ctis.inverters.AbstractInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    ):
        super().test__call__(
            a=a,
            images=images,
        )
