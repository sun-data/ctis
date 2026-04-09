from typing import Callable
import abc
import functools
import dataclasses
import astropy.units as u
import astropy.constants
import named_arrays as na

__all__ = [
    "AbstractInstrument",
    "AbstractLinearInstrument",
    "IdealInstrument",
]


ProjectionCallable = Callable[
    [na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]],
    na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
]


@dataclasses.dataclass
class AbstractInstrument(
    abc.ABC,
):
    """
    An interface describing a general CTIS instrument.

    The most important method of this interface is :meth:`image`,
    which represents the forward model of the instrument and maps the
    spectral radiance of the skyplane to detector counts.

    The other important method of this interface is :meth:`deproject`,
    which is the transpose of image and maps detector counts from an
    observed image to the corresponding spectral radiance on the skyplane.
    """

    @abc.abstractmethod
    def image(
        self,
        scene: na.AbstractScalar,
        noise: bool = True,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        r"""
        The forward model of this CTIS instrument, which maps spectral radiance
        on the skyplane to photons measured by the instrument's sensor.

        This method does `not` sum over wavelength, the result still has a
        wavelength axis so that this method is directly invertible.

        Parameters
        ----------
        scene
            The spectral radiance of an observed scene,
            evaluated on :attr:`coordinates_scene`,
            in units equivalent to
            :math:`\text{erg} \, \text{cm}^{-2} \, \text{sr}^{-1} \, \AA^{-1} \, \text{s}^{-1}`.
        noise
            Whether to include the effect of noise in the final image.
        """

    @abc.abstractmethod
    def backproject(
        self,
        image: na.AbstractScalar,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The inverse of the forward model of this CTIS instrument.
        Since the forward model, :meth:`image`, does not sum over wavelength,
        :meth:`backproject` can find the true inverse.

        Parameters
        ----------
        image
            A series of images captured by a CTIS instrument,
            evaluated on :attr:`coordinates_sensor`,
            in units of photons.
        """

    @property
    @abc.abstractmethod
    def coordinates_scene(self) -> na.AbstractSpectralPositionalVectorArray:
        """
        A grid of wavelength and position coordinates on the skyplane
        which will be used to construct the inverted scene.

        Normally the pitch of this grid is chosen to be the average
        plate scale of the instrument.
        """

    @property
    @abc.abstractmethod
    def coordinates_sensor(self) -> na.AbstractSpectralPositionalVectorArray:
        """
        A grid of wavelength and position coordinates on the detector plane.
        """

    @property
    @abc.abstractmethod
    def axis_wavelength(self) -> str:
        """
        The logical axis of :attr:`coordinates_scene` and :attr:`coordinates_sensor`
        corresponding to changing wavelength coordinate.
        """

    @property
    @abc.abstractmethod
    def axis_scene_xy(self) -> tuple[str, str]:
        """
        The logical axes of :attr:`coordinates_scene` corresponding to
        changing position coordinate.
        """

    @property
    @abc.abstractmethod
    def axis_sensor_xy(self) -> tuple[str, str]:
        """
        The logical axes of :attr:`coordinates_sensor` corresponding to
        changing position coordinate.
        """


@dataclasses.dataclass
class AbstractLinearInstrument(
    AbstractInstrument,
):
    """
    An instrument where the forward model can be represented using
    matrix multiplication.
    """

    @property
    @abc.abstractmethod
    def weights(
        self,
    ) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        The contribution of each voxel on the skyplane to each pixel on the
        detector.
        """

    @property
    @abc.abstractmethod
    def weights_transpose(
        self,
    ) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        The contribution of each pixel on the detector to each voxel on the
        skyplane.
        """

    @property
    def _volume_scene(self) -> na.AbstractScalar:
        """
        The volume of each voxel in :attr:`coordinates_scene`.
        """
        coords = self.coordinates_scene

        dw = coords.wavelength.volume_cell(self.axis_wavelength)

        dA = coords.position.volume_cell(self.axis_scene_xy)
        dA = na.as_named_array(dA)
        dA = dA.cell_centers(self.axis_wavelength)

        dV = dw * dA

        return dV

    @property
    def _energy_per_photon(self) -> u.Quantity | na.AbstractScalar:
        """
        The energy per photon for each wavelength of the scene.
        """

        h = astropy.constants.h
        c = astropy.constants.c

        w = self.coordinates_scene.wavelength.cell_centers(self.axis_wavelength)

        energy_per_photon = h * c / w / u.photon

        return energy_per_photon

    def image(
        self,
        scene: na.AbstractScalar,
        noise: bool = True,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        values_input = scene * self._volume_scene

        values_input = values_input / self._energy_per_photon

        values_input = values_input.to(u.photon)

        if noise:
            values_input = na.random.poisson(values_input)

        return na.FunctionArray(
            inputs=self.coordinates_sensor,
            outputs=na.regridding.regrid_from_weights(
                *self.weights,
                values_input=values_input,
            ),
        )

    def backproject(
        self,
        image: na.AbstractScalar,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        values_input = image * self._energy_per_photon

        values_input = values_input.to(u.erg)

        values_input = values_input / self._volume_scene

        return na.FunctionArray(
            inputs=self.coordinates_scene,
            outputs=na.regridding.regrid_from_weights(
                *self.weights_transpose,
                values_input=values_input,
            ),
        )


@dataclasses.dataclass
class IdealInstrument(
    AbstractLinearInstrument,
):
    """
    An idealized CTIS instrument which has a perfect point-spread function
    and no noise.
    """

    area_effective: u.Quantity | na.AbstractScalar
    r"""
    The effective area of the instrument aperture in units equivalent to
    :math:`\text{cm}^2`.
    """

    timedelta_exposure: u.Quantity | na.AbstractScalar
    r"""
    The exposure time of the instrument in units equivalent to :math:`\text{s}.
    """

    plate_scale: u.Quantity | na.AbstractScalar
    r"""The spatial scale of the image on the sensor in :math:`\text{arcsec} \,\text{pix}^-1`"""

    dispersion: u.Quantity | na.AbstractScalar
    r"""The magnitude of the dispersion in :math:`\text{m \AA} \,\text{pix}^-1`"""

    angle: u.Quantity | na.AbstractScalar
    """The angle of the dispersion direction with respect to the scene."""

    wavelength_ref: u.Quantity | na.AbstractScalar
    """
    The reference wavelength at which the center of the FOV lands at :attr:`position_ref`
    """

    position_ref: u.Quantity | na.AbstractScalar
    """
    The position where the reference wavelength is designed to land.
    """

    coordinates_scene: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    """
    A grid of wavelength and position coordinates on the skyplane
    which will be used to construct the inverted scene.

    Normally the pitch of this grid is chosen to be the average
    plate scale of the instrument.
    """

    coordinates_sensor: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    """
    A grid of wavelength and position coordinates on the detector plane.
    """

    axis_wavelength: str = dataclasses.MISSING
    """
    The logical axis of :attr:`coordinates_scene` and :attr:`coordinates_sensor`
    corresponding to changing wavelength coordinate.
    """

    axis_scene_xy: tuple[str, str] = dataclasses.MISSING
    """
    The logical axes of :attr:`coordinates_scene` corresponding to
    changing position coordinate.
    """

    axis_sensor_xy: tuple[str, str] = dataclasses.MISSING
    """
    The logical axes of :attr:`coordinates_sensor` corresponding to
    changing position coordinate.
    """

    def distortion(self, coordinates: na.SpectralPositionalVectorArray):
        unit_wavelength = coordinates.wavelength.unit
        rot = na.Cartesian2dRotationMatrixArray(self.angle)
        rot_grid = na.SpectralPositionalVectorArray(
            wavelength=coordinates.wavelength - self.wavelength_ref,
            position=rot @ coordinates.position,
        )
        disperse = na.SpectralPositionalMatrixArray(
            wavelength=na.SpectralPositionalVectorArray(
                wavelength=1,
                position=na.Cartesian2dVectorArray(
                    x=0 * unit_wavelength / u.arcsec,
                    y=0 * unit_wavelength / u.arcsec,
                ),
            ),
            position=na.Cartesian2dMatrixArray(
                x=na.SpectralPositionalVectorArray(
                    wavelength=1 / self.dispersion,
                    position=na.Cartesian2dVectorArray(
                        x=1 / self.plate_scale,
                        y=0 * u.pix / u.arcsec,
                    ),
                ),
                y=na.SpectralPositionalVectorArray(
                    wavelength=0 * u.pix / unit_wavelength,
                    position=na.Cartesian2dVectorArray(
                        x=0 * u.pix / u.arcsec,
                        y=1 / self.plate_scale,
                    ),
                ),
            ),
        )
        projected_grid = disperse @ rot_grid
        return na.SpectralPositionalVectorArray(
            wavelength=coordinates.wavelength,
            position=projected_grid.position + self.position_ref,
        )

    @functools.cached_property
    def weights(self) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:

        coordinates_input = self.distortion(self.coordinates_scene)
        coordinates_output = self.coordinates_sensor

        coordinates_input = coordinates_input.cell_centers(self.axis_wavelength)
        coordinates_output = coordinates_output.cell_centers(self.axis_wavelength)

        return na.regridding.weights(
            coordinates_input=coordinates_input.position,
            coordinates_output=coordinates_output.position,
            axis_input=self.axis_scene_xy,
            axis_output=self.axis_sensor_xy,
            method="conservative",
        )

    @functools.cached_property
    def weights_transpose(self):
        return na.regridding.transpose_weights(self.weights)

    def image(
        self,
        scene: na.AbstractScalar,
        noise: bool = True,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        scene = scene * self.area_effective * self.timedelta_exposure

        return super().image(scene)

    def backproject(
        self,
        image: na.AbstractScalar,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        result = super().backproject(image)

        result = result / (self.area_effective * self.timedelta_exposure)

        return result
