import numpy as np
from typing import Callable, ClassVar
import abc
import math
import functools
import dataclasses
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
from ._operators import Operators, aligned, framed, frames, regridder, vertices

__all__ = [
    "AbstractInstrument",
    "AbstractLinearInstrument",
    "IdealInstrument",
    "OptikaInstrument",
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
        scene: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool = True,
        noise: bool = True,
        uncertainty: bool = False,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        r"""
        The forward model of this CTIS instrument, which maps spectral radiance
        on the skyplane to the electrons measured by the instrument's sensor.

        Parameters
        ----------
        scene
            The spectral radiance of an observed scene,
            evaluated on :attr:`coordinates_scene`,
            in units equivalent to
            :math:`\text{erg} \, \text{cm}^{-2} \, \text{sr}^{-1} \, \AA^{-1} \, \text{s}^{-1}`.
        integrate
            Whether to integrate along the wavelength axis.
            A real CTIS instrument integrates along wavelength,
            but sometimes it's useful to keep the wavelengths separate
            for demonstration purposes.
        noise
            Whether to include the effect of noise in the final image.
        uncertainty
            Whether to attach the standard deviation of the measurement noise
            to the result, as a
            :class:`~named_arrays.NormalUncertainScalarArray`.
            The variance is computed for each wavelength *before* the
            integration along the wavelength axis and summed in quadrature, so
            it is exact even for the integrated image.
        """

    @abc.abstractmethod
    def backproject(
        self,
        image: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool = True,
        unit: None | u.UnitBase = None,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The backward model of this CTIS instrument, which maps photons measured
        by the sensor to spectral radiance on the skyplane.

        This is the complementary operation to :meth:`image`, but it is not
        an inverse of :meth:`image`, this method will spread out the photons
        from each pixel evenly across the voxels in the scene that `could` have
        contributed to the measured signal.

        Parameters
        ----------
        image
            A series of images captured by a CTIS instrument,
            evaluated on :attr:`coordinates_sensor`,
            in units of photons.
        integrate
            Complement of the `integrate` keyword of :meth:`image`.
        unit
            The unit of the backprojected spectral radiance.
            The forward model, :meth:`image`, accepts a scene in either photon
            or energy units, so the backprojection is expressed in whichever the
            caller requests, converting between photon and energy units using
            the energy per photon.
            If :obj:`None` (the default), the radiance is left in the natural
            units of the backprojection and is not converted.
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
    def channel(self):
        """
        Human-readable name of each independent CTIS channel.
        """

    @property
    @abc.abstractmethod
    def axis_channel(self) -> str | tuple[str, ...]:
        """
        The logical axis or axes of this instrument corresponding to
        the different CTIS channels.
        """

    @property
    @abc.abstractmethod
    def axis_wavelength(self) -> str:
        """
        The logical axis of :attr:`coordinates_scene` and :attr:`coordinates_sensor`
        that corresponds to changing wavelength coordinate.
        """

    @property
    @abc.abstractmethod
    def axis_line(self) -> None | str:
        """
        The logical axis of :attr:`coordinates_scene` which separates the
        spectral lines of the scene, or :obj:`None` if the scene is one band
        of wavelengths.

        Each line has its own wavelengths along :attr:`axis_wavelength`, the
        same number for every line, and all the lines are imaged onto the
        same sensors.
        An image which is integrated along wavelength is therefore summed
        over the lines too, and the backprojection of one is spread over the
        wavelengths of every line.
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

    @property
    @abc.abstractmethod
    def num_channel(self) -> int:
        """
        The total number of dispersion magnitudes/angles observed by this
        instrument.
        """

    @property
    def _axis_channel(self) -> tuple[str, ...]:
        """
        :attr:`axis_channel` as a tuple of axes, whether it names one axis
        or several.
        """
        axis_channel = self.axis_channel
        if isinstance(axis_channel, str):
            axis_channel = (axis_channel,)
        return tuple(axis_channel)

    @property
    def _axis_spectral(self) -> tuple[str, ...]:
        """
        The axes that an image which is integrated along wavelength is summed
        over: :attr:`axis_line`, if the scene has several lines, and
        :attr:`axis_wavelength`.
        """
        axis_line = self.axis_line
        if axis_line is None:
            return (self.axis_wavelength,)
        return (axis_line, self.axis_wavelength)

    def _operators(self, dtype: type = np.float64) -> None | Operators:
        """
        This instrument as a pair of sparse matrices which act on plain
        arrays, or :obj:`None` if it does not provide them.

        The inverters iterate on these, in place of :meth:`image` and
        :meth:`backproject`, if they are provided.

        Parameters
        ----------
        dtype
            The precision of the matrices and of the arrays they act on,
            :obj:`numpy.float64` or :obj:`numpy.float32`.
        """
        return None


@dataclasses.dataclass
class AbstractLinearInstrument(
    AbstractInstrument,
):
    """
    An instrument where the forward model can be represented using
    matrix multiplication.
    """

    _operators_kept: None | tuple[np.dtype, tuple, None | Operators] = (
        dataclasses.field(
            default=None,
            init=False,
            repr=False,
            compare=False,
        )
    )
    """
    The operators of this instrument as :meth:`_operators` last assembled
    them, with the precision they are in and the fields of this instrument
    that their factors were computed from.
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
    def num_channel(self) -> int:

        # operators which have been assembled hold the count, and have
        # released the weights, which counting on them would compute again
        kept = self._operators_kept
        if kept is not None and kept[2] is not None:
            return kept[2].num_channel

        shape = self.weights[0].shape

        num_channels = 1
        for ax in self._axis_channel:
            num_channels = num_channels * shape[ax]

        return num_channels

    @property
    def _fields_factors(self) -> tuple:
        """
        The fields of this instrument which the factors around the matrices
        of its operators are computed from, and which the matrices do not
        depend on.

        If any of them is assigned, the factors are computed again, on the
        same matrices.
        """
        return ()

    @classmethod
    def _overridden(cls) -> bool:
        """
        Whether this class replaces any of the methods which its operators
        stand in for.

        The operators are the forward model, the noise model and the
        backprojection of the class which defines their factors,
        :meth:`_factors`, so a subclass of that class which defines any of
        those otherwise does not provide them, and is imaged, backprojected
        and inverted with its own methods.
        """
        owner = next(c for c in cls.__mro__ if "_factors" in vars(c))
        names = (
            "image",
            "backproject",
            "_image_expected",
            "_image_matrices",
            "_image_weights",
            "_backproject_matrices",
            "_backproject_weights",
            "_shot_noise",
            "weights_transpose",
        )
        return any(
            getattr(cls, name, None) is not getattr(owner, name, None) for name in names
        )

    def _operators(self, dtype: type = np.float64) -> None | Operators:
        """
        This instrument as a pair of sparse matrices which act on plain
        arrays, or :obj:`None` if it cannot be expressed as one.

        The matrices are kept, in the precision they were last asked for in,
        so they are assembled once for any number of inversions in that
        precision, and again if another is asked for.

        The matrices hold what :attr:`weights` and :attr:`weights_transpose`
        hold, in another form, and an instrument large enough for it to
        matter cannot afford both.
        Assembling the matrices therefore releases the weights,
        which are computed again if they are needed.

        The factors around the matrices follow the fields of this instrument
        that they are computed from, :attr:`_fields_factors`: if any of
        those has been assigned since the operators were assembled, the
        factors are computed again, on the same matrices.
        The matrices themselves are kept for as long as :attr:`weights` would
        be.

        A subclass which replaces a method that the operators stand in for
        does not provide them, see :meth:`_overridden`.

        Parameters
        ----------
        dtype
            The precision of the matrices and of the arrays they act on,
            :obj:`numpy.float64` or :obj:`numpy.float32`.
        """
        dtype = np.dtype(dtype)

        if self._overridden():
            return None

        fields = self._fields_factors

        kept = self._operators_kept
        if kept is not None and kept[0] == dtype:
            _, fields_kept, operators = kept

            if all(a is b for a, b in zip(fields, fields_kept)):
                return operators

            # only the factors depend on those fields, so the matrices,
            # which are the expensive part, are kept
            if operators is not None:
                operators = self._operators_refreshed(operators)
                self._operators_kept = (dtype, fields, operators)
                return operators

        # matrices kept in another precision are released first, so that the
        # two never have to fit in memory together
        del kept
        self._operators_kept = None

        operators = self._operators_new(dtype)

        self._operators_kept = (dtype, fields, operators)

        return operators

    def _operators_refreshed(self, operators: Operators) -> None | Operators:
        """
        Operators on the same matrices, with the factors around them computed
        again from the fields of this instrument, or :obj:`None` if the
        fields can no longer be expressed as those factors.

        Parameters
        ----------
        operators
            The operators whose matrices to keep.
        """
        factors = self._factors(
            shape_channel=operators.shape_channel,
            shape_line=operators.shape_line,
            shape_scene=operators.shape_scene,
            shape_sensor=operators.shape_sensor,
            dtype=operators.dtype,
        )
        if factors is None:
            return None

        result = dataclasses.replace(operators, **factors)

        # the matrix of the backprojection does not depend on the factors
        result._backward = operators._backward

        return result

    def _factors(
        self,
        shape_channel: dict[str, int],
        shape_line: dict[str, int],
        shape_scene: dict[str, int],
        shape_sensor: dict[str, int],
        dtype: np.dtype,
    ) -> None | dict[str, np.ndarray | u.UnitBase]:
        """
        The factors around the matrices of the operators of this instrument,
        and the units of the arrays they act on, as the fields of
        :class:`~ctis.instruments._operators.Operators` of the same names,
        or :obj:`None` if this instrument cannot be expressed as those
        factors.

        An instrument provides operators by defining this method,
        :meth:`_grids_weights` and, if :attr:`weights` weighs the cells of
        the scene, :meth:`_weights_input`.
        By default it provides none.

        Parameters
        ----------
        shape_channel
            The axes of the channels and their lengths.
        shape_line
            The axis of the spectral lines and its length, if the scene has
            several.
        shape_scene
            The axes of a scene and their lengths.
        shape_sensor
            The two spatial axes of an image and their lengths.
        dtype
            The precision of the factors.
        """
        return None

    def _grids_weights(
        self,
    ) -> tuple[na.AbstractCartesian2dVectorArray, na.AbstractCartesian2dVectorArray]:
        """
        The vertices of the cells of the scene and of the pixels of the
        sensor which :attr:`weights` was computed from, both on the plane of
        the sensor.
        """
        raise NotImplementedError

    def _weights_input(self) -> None | na.AbstractScalar:
        """
        The weight which :attr:`weights` gives every cell of the scene, for
        each channel and each wavelength, beyond the fraction of the cell
        which lands in each pixel, or :obj:`None` if it gives none.

        The backprojection inverts this weight, as
        :func:`regridding.transpose_weights_conservative` does.
        """
        return None

    def _operators_new(self, dtype: np.dtype) -> None | Operators:
        """
        Assemble the operators of this instrument, see :meth:`_operators`.

        Parameters
        ----------
        dtype
            The precision of the matrices and of the arrays they act on.
        """

        axis_channel = self._axis_channel
        axis_wavelength = self.axis_wavelength
        axis_line = self._axis_spectral[:~0]

        weights = self.weights
        array, shape_grid_scene, shape_grid_sensor = weights

        # an instrument which varies along any other axis, such as time,
        # is left to the named arrays
        if set(array.shape) != {*axis_channel, *axis_line, axis_wavelength}:
            return None

        shape_channel = {ax: array.shape[ax] for ax in axis_channel}
        shape_line = {ax: array.shape[ax] for ax in axis_line}
        shape_wavelength = {axis_wavelength: array.shape[axis_wavelength]}
        shape_scene_xy = {
            ax: shape_grid_scene[ax] for ax in shape_grid_scene if ax not in array.shape
        }
        shape_sensor = {
            ax: shape_grid_sensor[ax]
            for ax in shape_grid_sensor
            if ax not in array.shape
        }
        shape_scene = shape_line | shape_wavelength | shape_scene_xy
        shape_voxels = shape_channel | shape_scene

        factors = self._factors(
            shape_channel=shape_channel,
            shape_line=shape_line,
            shape_scene=shape_scene,
            shape_sensor=shape_sensor,
            dtype=dtype,
        )
        if factors is None:
            return None

        # the vertices of the two grids that the weights were computed from
        position_scene, position_sensor = self._grids_weights()
        unit_position = na.unit_normalized(position_sensor.x)
        vertices_scene = vertices(
            coordinates=position_scene,
            shape=shape_channel
            | shape_line
            | shape_wavelength
            | {ax: num + 1 for ax, num in shape_scene_xy.items()},
            unit=unit_position,
        )
        vertices_sensor = vertices(
            coordinates=position_sensor,
            shape=shape_channel
            | shape_line
            | shape_wavelength
            | {ax: num + 1 for ax, num in shape_sensor.items()},
            unit=unit_position,
        )

        # the weight of every cell of the scene, if the weights have one
        weights_scene = self._weights_input()
        if weights_scene is not None:
            weights_scene = aligned(
                a=weights_scene,
                shape=shape_voxels,
                unit=u.dimensionless_unscaled,
            )
            weights_scene = np.ascontiguousarray(
                np.broadcast_to(weights_scene, tuple(shape_voxels.values())),
            )

        # the matrix takes a scene for each channel, rather than one shared
        # by the channels, so that its transpose gives a backprojection
        # for each channel, and it sums the images over wavelength, but not
        # over the lines, which can each have a response of their own
        forward = regridder(
            weights=weights,
            shape_input=shape_voxels,
            shape_output=shape_channel
            | shape_line
            | {axis_wavelength: 1}
            | shape_sensor,
            dtype=dtype,
        )

        # the weights are released, so that the matrix of the backprojection,
        # which is derived from the one above when it is first needed,
        # does not have to fit in memory beside them
        del weights, array
        vars(self).pop("weights", None)
        vars(self).pop("weights_transpose", None)

        return Operators(
            forward=forward,
            vertices_scene=vertices_scene,
            vertices_sensor=vertices_sensor,
            shape_channel=shape_channel,
            shape_scene=shape_scene,
            shape_sensor=shape_sensor,
            shape_line=shape_line,
            weights_scene=weights_scene,
            **factors,
        )

    @property
    def _volume_scene(self) -> na.AbstractScalar:
        """
        The volume of each voxel in :attr:`coordinates_scene`.
        """
        # `.spectral_positional` accepts a Doppler scene (as used by
        # `IdealInstrument`), which does not implement `volume_cell` directly.
        coords = self.coordinates_scene.spectral_positional

        return coords.volume_cell((self.axis_wavelength, *self.axis_scene_xy))

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

    def _collapse_wavelength(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.AbstractSpectralPositionalVectorArray:
        """
        Collapse the wavelength coordinates of an image to the band edges,
        which are the coordinates of an image integrated along the wavelength
        axis, and over the spectral lines if the scene has several.

        Parameters
        ----------
        coordinates
            The coordinates of the image before it is integrated.
        """
        axis = self._axis_spectral

        return coordinates.replace(
            wavelength=na.stack(
                arrays=[
                    coordinates.wavelength.min(axis),
                    coordinates.wavelength.max(axis),
                ],
                axis=self.axis_wavelength,
            )
        )

    def _image_expected(
        self,
        scene: na.AbstractScalar | na.AbstractFunctionArray,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The photons that a scene sends to every pixel at each wavelength,
        before they are counted: without noise, and negative wherever the
        scene makes them so.

        The forward model is linear up to this point.

        Parameters
        ----------
        scene
            The spectral radiance of the observed scene.
        """

        if isinstance(scene, na.AbstractFunctionArray):
            if not np.all(scene.inputs == self.coordinates_scene):
                raise ValueError(
                    "`scene.inputs` and `self.coordinates_scene` are not equal."
                )
            scene = scene.outputs

        values_input = scene * self._volume_scene

        values_input = values_input / self._energy_per_photon

        values_input = values_input.to(u.photon)

        values_output = na.regridding.regrid_from_weights(
            *self.weights,
            values_input=values_input,
        )

        return na.FunctionArray(
            inputs=self.coordinates_sensor,
            outputs=values_output,
        )

    def image(
        self,
        scene: na.AbstractScalar | na.AbstractFunctionArray,
        noise: bool = True,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        # this low-level forward model always returns the per-wavelength photons;
        # integration over wavelength (and any conversion to electrons) is done
        # by the concrete subclass's `image`.

        result = self._image_expected(scene)

        # the photons counted at a wavelength cannot be negative
        values_output = np.maximum(result.outputs, 0)

        if noise:
            values_output = na.random.poisson(values_output)

        return na.FunctionArray(
            inputs=result.inputs,
            outputs=values_output,
        )

    def backproject(
        self,
        image: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool = True,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        if isinstance(image, na.AbstractFunctionArray):
            if not np.all(image.inputs.position == self.coordinates_sensor.position):
                raise ValueError(
                    "`image.inputs` and `self.coordinates_sensor` are not equal."
                )
            image = image.outputs

        coordinates = self.coordinates_scene

        # an integrated image is spread evenly over the wavelengths of every
        # line, so that the backprojection conserves its total
        num_wavelength = coordinates.wavelength.shape[self.axis_wavelength] - 1
        axis_line = self.axis_line
        if axis_line is not None:
            num_wavelength = num_wavelength * coordinates.wavelength.shape[axis_line]

        values_input = image

        if integrate:
            values_input = values_input / num_wavelength

        values_output = na.regridding.regrid_from_weights(
            *self.weights_transpose,
            values_input=values_input,
        )

        values_output = values_output * self._energy_per_photon

        values_output = values_output.to(u.erg)

        values_output = values_output / self._volume_scene

        return na.FunctionArray(
            inputs=coordinates,
            outputs=values_output,
        )


@dataclasses.dataclass
class IdealInstrument(
    AbstractLinearInstrument,
):
    """
    An idealized CTIS instrument model.

    This ideal instrument is characterized by an effective area,
    exposure time, plate scale and dispersion magnitude/angle.

    It has no distortion or vignetting, and it considers only photon shot
    noise and read noise.  It may have a point-spread function, :attr:`psf`.

    The images and the backprojections which are integrated along the
    wavelength axis are computed with a pair of sparse matrices, which are
    assembled from :attr:`weights` the first time they are needed, and which
    are the matrices that the inverters iterate on.
    The matrix of the forward model sums the photons over wavelength, so the
    shot noise of an image is drawn once for each pixel, from the photons
    expected there at all wavelengths together.
    A sum of independent Poisson variables is itself a Poisson variable, so
    this is the distribution that a draw for each wavelength would have.

    With `integrate` set to :obj:`False`, or if the instrument cannot be
    expressed as a pair of matrices, the images and the backprojections are
    computed from :attr:`weights` and :attr:`weights_transpose` instead, one
    wavelength at a time, with the shot noise drawn for each wavelength,
    which is slower.
    An instrument cannot be expressed as a pair of matrices if its
    :attr:`quantum_yield` varies along :attr:`axis_wavelength` or across the
    sensor, or if it varies along an axis of its own, such as time.
    The quantum yield can be different for each channel, and for each
    spectral line of a scene which has several, :attr:`axis_line`.
    Neither can a subclass which replaces :meth:`image`, :meth:`backproject`
    or one of the methods they are built from, which is left to its own
    methods.

    The photons expected in a pixel are kept from being negative once they
    are summed over wavelength, so the forward model is linear up to the
    point where the photons are counted, and a scene which is negative
    somewhere can cancel the light of one wavelength with that of another.
    The exception is an image with noise which is computed from
    :attr:`weights`: its photons are drawn for each wavelength, and each
    wavelength is kept from being negative before it is drawn.
    """

    area_effective: u.Quantity | na.AbstractScalar
    r"""
    The effective area of the instrument aperture in units equivalent to
    :math:`\text{cm}^2`.
    """

    timedelta_exposure: u.Quantity | na.AbstractScalar
    r"""
    The exposure time of the instrument in units equivalent to :math:`\text{s}`.
    """

    plate_scale: u.Quantity | na.AbstractScalar | na.Cartesian2dVectorArray
    r"""
    The spatial scale of the image on the sensor in units equivalent to
    :math:`\text{arcsec} \,\text{pix}^-1`
    """

    dispersion: u.Quantity | na.AbstractScalar
    r"""
    The magnitude of the spectral dispersion in units equivalent to 
    :math:`\text{m \AA} \,\text{pix}^-1`.
    """

    angle: u.Quantity | na.AbstractScalar
    """
    The angle of the dispersion direction with respect to the scene.
    """

    wavelength_ref: u.Quantity | na.AbstractScalar
    """
    The reference wavelength at which the center of the FOV lands at :attr:`position_ref`
    """

    position_ref: u.Quantity | na.AbstractScalar | na.Cartesian2dVectorArray
    """
    The position on the sensor where center of the FOV lands at the reference
    wavelength.
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
    A grid of wavelength and position coordinates on the sensor plane.
    """

    channel: str | na.AbstractScalar = dataclasses.MISSING
    """
    Human-readable name of each independent CTIS channel.
    """

    axis_channel: str | tuple[str, ...] = dataclasses.MISSING
    """
    The logical axis or axes of this instrument corresponding to
    the different dispersion magnitudes and angles.
    """

    axis_wavelength: str = dataclasses.MISSING
    """
    The logical axis of :attr:`coordinates_scene` and :attr:`coordinates_sensor`
    that corresponds to changing wavelength coordinate.
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

    quantum_yield: u.Quantity | na.AbstractScalar = 1 * u.electron / u.photon
    r"""
    The number of electrons generated in the sensor per incident photon, in
    units equivalent to :math:`\text{electron} \, \text{photon}^{-1}`.

    For this idealized instrument the quantum yield is a constant, so it can be
    applied after integrating over wavelength.
    """

    read_noise: u.Quantity | na.AbstractScalar = 0 * u.electron
    """
    The standard deviation of the Gaussian read noise added to each pixel
    once per readout (after integrating over wavelength), in electrons.
    """

    psf: None | na.AbstractScalar = None
    r"""
    The point-spread function of the instrument on the sensor, or :obj:`None`
    (the default) for none.

    It is the fraction of the light landing in a pixel which reaches each of
    the pixels around it, so it is the point-spread function convolved with a
    pixel twice, once for the pixel the light lands in and once for the pixel
    it is collected in.
    Its axes :attr:`axis_psf_xy` run along :attr:`axis_sensor_xy`, and are
    centered on the element at index :math:`\lfloor n / 2 \rfloor`.
    Any other axis is broadcast by name: :attr:`axis_channel` or
    :attr:`axis_wavelength` give a point-spread function for each channel or
    wavelength, and :attr:`axis_sensor_xy` one which varies across the sensor.

    The point-spread function is folded into :attr:`weights` with
    :func:`named_arrays.regridding.convolve_weights`, so :meth:`image` blurs the
    scene by it and :meth:`backproject`, the transpose, accounts for it too.
    """

    axis_psf_xy: tuple[str, str] = ("psf_x", "psf_y")
    """
    The logical axes of :attr:`psf` which run along :attr:`axis_sensor_xy`.
    """

    axis_line: None | str = None
    """
    The logical axis of :attr:`coordinates_scene` and
    :attr:`coordinates_sensor` which separates the spectral lines of the
    scene, or :obj:`None` (the default) if the scene is one band of
    wavelengths.

    Each line has its own wavelengths along :attr:`axis_wavelength`, the same
    number for every line, and all the lines are imaged onto the same
    sensors.
    An image which is integrated along wavelength is therefore summed over
    the lines too, and the backprojection of one is spread over the
    wavelengths of every line.
    The :attr:`quantum_yield` can be different for each line.
    """

    @property
    def _fields_factors(self) -> tuple:
        """
        The fields of this instrument which the factors around the matrices
        of its operators are computed from, and which the matrices do not
        depend on.
        """
        return (
            self.area_effective,
            self.timedelta_exposure,
            self.quantum_yield,
            self.read_noise,
        )

    def _shot_noise(self, image: na.ScalarArray) -> na.ScalarArray:
        # photon shot noise, converted back into electrons to match the
        # electron-valued image
        photons = image / self.quantum_yield
        uncertainty = np.sqrt(photons.to_value(u.ph)) * u.ph
        return uncertainty * self.quantum_yield

    def distortion(self, coordinates: na.SpectralPositionalVectorArray):
        """
        A linear mapping between skyplane coordinates and sensor coordinates.

        Parameters
        ----------
        coordinates
            Grid of spatial/spectral coordinates on the skyplane.
        """
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
    def _coordinates_input(self) -> na.AbstractSpectralPositionalVectorArray:
        """
        The :attr:`coordinates_scene` property mapped onto the sensor plane and
        transformed onto cell centers along the wavelength axis.
        """

        coordinates_input = self.distortion(self.coordinates_scene)

        coordinates_input = coordinates_input.cell_centers(self.axis_wavelength)

        return coordinates_input

    @functools.cached_property
    def _coordinates_output(self) -> na.AbstractSpectralPositionalVectorArray:
        """
        The :attr:`coordinates_sensor` perturbed to avoid degeneracies while
        resampling and transformed onto cell centers along the wavelength axis.
        """

        coordinates_output = self.coordinates_sensor

        coordinates_output = coordinates_output.cell_centers(self.axis_wavelength)

        return coordinates_output

    @functools.cached_property
    def weights(self) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:

        coordinates_input = self._coordinates_input
        coordinates_output = self._coordinates_output

        weights = na.regridding.weights(
            coordinates_input=coordinates_input.position,
            coordinates_output=coordinates_output.position,
            axis_input=self.axis_scene_xy,
            axis_output=self.axis_sensor_xy,
            method="conservative",
        )

        if self.psf is not None:
            weights = na.regridding.convolve_weights(
                weights=weights,
                kernel=self.psf,
                axis=dict(zip(self.axis_sensor_xy, self.axis_psf_xy)),
            )

        return weights

    @functools.cached_property
    def weights_transpose(self):

        coordinates_input = self._coordinates_input
        coordinates_output = self._coordinates_output

        return na.regridding.transpose_weights_conservative(
            weights=self.weights,
            coordinates_input=coordinates_input.position,
            coordinates_output=coordinates_output.position,
            axis_input=self.axis_scene_xy,
            axis_output=self.axis_sensor_xy,
        )

    @property
    def num_channel(self) -> int:

        # the channels are counted on the grids that the weights are computed
        # from, so that counting them does not compute the weights.
        # Only the axes of the channels are compared, since along those of
        # the sensor the grids hold vertices where a point-spread function
        # holds cells.
        arrays = (
            self._coordinates_input.position,
            self._coordinates_output.position,
            self.psf,
        )

        shape = dict()
        for a in arrays:
            for ax, num in na.shape(a).items():
                shape[ax] = max(num, shape.get(ax, 1))

        return math.prod(shape[ax] for ax in self._axis_channel)

    def _factors(
        self,
        shape_channel: dict[str, int],
        shape_line: dict[str, int],
        shape_scene: dict[str, int],
        shape_sensor: dict[str, int],
        dtype: np.dtype,
    ) -> None | dict[str, np.ndarray | u.UnitBase]:
        """
        The factors around the matrices of the operators of this instrument,
        and the units of the arrays they act on, as the fields of
        :class:`~ctis.instruments._operators.Operators` of the same names,
        or :obj:`None` if the fields of this instrument cannot be expressed
        as those factors.

        Parameters
        ----------
        shape_channel
            The axes of the channels and their lengths.
        shape_line
            The axis of the spectral lines and its length, if the scene has
            several.
        shape_scene
            The axes of a scene and their lengths.
        shape_sensor
            The two spatial axes of an image and their lengths.
        dtype
            The precision of the factors.
        """
        shape_voxels = shape_channel | shape_scene
        shape_pixels = shape_channel | shape_sensor

        exposure = self.area_effective * self.timedelta_exposure
        volume = self._volume_scene
        energy = self._energy_per_photon

        # an image is spread over the wavelengths of every line
        num_wavelength = shape_scene[self.axis_wavelength]
        num_wavelength = num_wavelength * math.prod(shape_line.values())

        # the units that `image` and `backproject` give their results in
        unit_images = u.photon * na.unit_normalized(self.quantum_yield)
        unit_scene = u.erg / na.unit_normalized(volume) / na.unit_normalized(exposure)

        # the factors that `image` applies to the scene before resampling it,
        # and those that `backproject` applies around its resampling
        scale = (1 * unit_scene) * exposure * volume / energy
        scale_backproject = (1 * unit_images) / self.quantum_yield / num_wavelength
        scale_backproject = scale_backproject * energy / volume / exposure

        scale = aligned(scale, shape=shape_voxels, unit=u.photon)
        scale_backproject = aligned(
            a=scale_backproject,
            shape=shape_voxels,
            unit=unit_scene,
        )
        gain = aligned(
            a=self.quantum_yield,
            shape=shape_channel | shape_line,
            unit=unit_images / u.photon,
        )
        read_noise = aligned(self.read_noise, shape=shape_pixels, unit=unit_images)

        # a quantum yield which varies with wavelength or across the sensor
        # cannot be applied after the sum over wavelength, and anything
        # with other axes or units is left to the named arrays
        if any(a is None for a in (scale, scale_backproject, gain, read_noise)):
            return None

        # the factors of the voxels are given one more axis, which the
        # frames are broadcast along, and every factor the precision of the
        # matrices.
        # The variance of the shot noise of the photons, in units of the
        # images, is their signal times the quantum yield
        return dict(
            scale=scale[..., np.newaxis].astype(dtype),
            scale_backproject=scale_backproject[..., np.newaxis].astype(dtype),
            gain=gain.astype(dtype),
            vmr=gain.astype(dtype),
            variance_read=np.square(read_noise).astype(dtype),
            unit_scene=unit_scene,
            unit_images=unit_images,
        )

    def _grids_weights(
        self,
    ) -> tuple[na.AbstractCartesian2dVectorArray, na.AbstractCartesian2dVectorArray]:
        return (
            self._coordinates_input.position,
            self._coordinates_output.position,
        )

    def image(
        self,
        scene: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool = True,
        noise: bool = True,
        uncertainty: bool = False,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        # the matrices sum over wavelength, so they only give the image
        # which is integrated along it
        if integrate:
            result = self._image_matrices(
                scene=scene,
                noise=noise,
                uncertainty=uncertainty,
            )
            if result is not None:
                return result

        return self._image_weights(
            scene=scene,
            integrate=integrate,
            noise=noise,
            uncertainty=uncertainty,
        )

    def _image_matrices(
        self,
        scene: na.AbstractScalar | na.AbstractFunctionArray,
        noise: bool,
        uncertainty: bool,
    ) -> None | na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The image of a scene, integrated along the wavelength axis, computed
        with the sparse matrices of this instrument, or :obj:`None` if the
        instrument or the scene cannot be expressed as plain arrays, which
        leaves the image to :meth:`_image_weights`.

        Any axes of the scene beyond its own are those of independent frames,
        which are imaged in the same pass over the matrix.

        The shot noise is drawn once for each pixel, from the photons
        expected there at all wavelengths together, where
        :meth:`_image_weights` draws it for each wavelength.
        The two have the same distribution, since a sum of independent
        Poisson variables is itself a Poisson variable, but not the same
        values for a given seed.

        Parameters
        ----------
        scene
            The spectral radiance of the observed scene.
        noise
            Whether to include the effect of noise in the image.
        uncertainty
            Whether to attach the standard deviation of the noise to the
            image.
        """
        operators = self._operators()
        if operators is None:
            return None

        if isinstance(scene, na.AbstractFunctionArray):
            if not np.all(scene.inputs == self.coordinates_scene):
                raise ValueError(
                    "`scene.inputs` and `self.coordinates_scene` are not equal."
                )
            scene = scene.outputs

        shape_images = operators.shape_images

        # every other axis of the scene is an axis of the frames, unless it
        # is one of those of the images, which the named arrays are left to
        shape_frames = frames(
            shape=na.shape(scene),
            shape_own=operators.shape_scene,
            shape_other=shape_images,
        )
        if shape_frames is None:
            return None

        values = framed(
            a=scene,
            shape=operators.shape_scene,
            shape_frames=shape_frames,
            unit=operators.unit_scene,
        )
        if values is None:
            return None

        shape_line = operators.shape_line
        shape_photons = operators.shape_channel | shape_line | operators.shape_sensor
        dims_frames = tuple(shape_frames.values())

        if noise:
            # the photons expected in every pixel at all the wavelengths of
            # each line together, which are drawn for each line, since
            # each line can have a quantum yield of its own
            expected = operators.photons(values)
            expected = expected.reshape(expected.shape[:~0] + dims_frames)
            expected = na.ScalarArray(
                ndarray=expected << u.photon,
                axes=(*shape_photons, *shape_frames),
            )
            photons = na.random.poisson(expected)

            # convert the measured photons into electrons
            electrons = photons * self.quantum_yield
            if shape_line:
                electrons = electrons.sum(tuple(shape_line))

            # add read noise once, at the readout
            electrons = na.random.normal(loc=electrons, scale=self.read_noise)

            if uncertainty:
                # the width of the shot noise is that of the photons
                # expected, whatever the noise drawn above
                width = self._shot_noise(expected * self.quantum_yield)
                variance = np.square(width)
                if shape_line:
                    variance = variance.sum(tuple(shape_line))
                width = np.sqrt(variance + np.square(self.read_noise))

        else:
            # the images expected, and the variance of their noise
            images, variance = operators.image(values)

            images = images.reshape(images.shape[:~0] + dims_frames)
            electrons = na.ScalarArray(
                ndarray=images << operators.unit_images,
                axes=(*shape_images, *shape_frames),
            )

            if uncertainty:
                variance = variance.reshape(variance.shape[:~0] + dims_frames)
                width = na.ScalarArray(
                    ndarray=np.sqrt(variance) << operators.unit_images,
                    axes=(*shape_images, *shape_frames),
                )

        if uncertainty:
            electrons = na.NormalUncertainScalarArray(nominal=electrons, width=width)

        return na.FunctionArray(
            inputs=self._collapse_wavelength(self.coordinates_sensor),
            outputs=electrons,
        )

    def _image_weights(
        self,
        scene: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool,
        noise: bool,
        uncertainty: bool,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The image of a scene, computed from :attr:`weights`, one wavelength
        at a time, as named arrays.

        This is the image of any instrument and any scene, and the only one
        which keeps the wavelengths separate.

        The photons of a pixel are kept from being negative where they are
        counted: once they are summed over wavelength, if the image is
        integrated along it, as :meth:`_image_matrices` does, and for each
        wavelength otherwise.
        With noise, the photons of each wavelength are drawn on their own,
        and a draw needs an expectation which is not negative, so there each
        wavelength is kept from being negative before the sum.
        The difference only shows for a scene which is negative somewhere.

        Parameters
        ----------
        scene
            The spectral radiance of the observed scene.
        integrate
            Whether to integrate along the wavelength axis.
        noise
            Whether to include the effect of noise in the image.
        uncertainty
            Whether to attach the standard deviation of the noise to the
            image.
        """

        scene = scene * self.area_effective * self.timedelta_exposure

        # an integrated image is summed over wavelength, and over the lines
        # of the scene if it has several
        axis = self._axis_spectral

        # keep the wavelength axis so the uncertainty can be computed for each
        # wavelength before integrating.
        if integrate and not noise:
            result = self._image_expected(scene)
        else:
            result = super().image(
                scene=scene,
                noise=noise,
            )

        # convert the measured photons into electrons
        electrons = result.outputs * self.quantum_yield
        coordinates = result.inputs

        if uncertainty:
            # the shot-noise width uses the *expected* electrons; recompute the
            # noiseless image when `noise` replaced them with a realization.
            if not noise:
                expected = electrons
            elif integrate:
                expected = self._image_expected(scene).outputs * self.quantum_yield
            else:
                expected = (
                    super().image(scene=scene, noise=False).outputs * self.quantum_yield
                )

            if integrate:
                # the variance of each wavelength is given the sign of the
                # electrons expected there, so that the variances are summed
                # the way the electrons are, and kept from being negative
                # afterward
                variance = np.square(self._shot_noise(np.abs(expected)))
                variance = (np.sign(expected) * variance).sum(axis)
                variance = np.maximum(variance, 0)

                # add read noise once, at the integrated readout
                width = np.sqrt(variance + np.square(self.read_noise))
            else:
                width = self._shot_noise(expected)

        if integrate:
            electrons = electrons.sum(axis)
            coordinates = self._collapse_wavelength(coordinates)

            if noise:
                # add read noise once, at the integrated readout
                electrons = na.random.normal(loc=electrons, scale=self.read_noise)
            else:
                # the electrons counted in a pixel cannot be negative
                electrons = np.maximum(electrons, 0)

        if uncertainty:
            electrons = na.NormalUncertainScalarArray(nominal=electrons, width=width)

        return na.FunctionArray(
            inputs=coordinates,
            outputs=electrons,
        )

    def _to_unit(
        self,
        radiance: na.AbstractScalar,
        unit: None | u.UnitBase,
    ) -> na.AbstractScalar:
        """
        Express the (energy) backprojected radiance in `unit`, dividing by the
        energy per photon (:attr:`_energy_per_photon`) when `unit` is a photon
        rather than an energy unit. A `unit` compatible with neither raises a
        :class:`~astropy.units.UnitConversionError`.
        """
        if unit is None:
            return radiance

        if unit.is_equivalent(na.unit_normalized(radiance)):
            return radiance.to(unit)

        # the natural (energy) radiance is divided by the energy per photon to
        # express it in photon units.
        return (radiance / self._energy_per_photon).to(unit)

    def backproject(
        self,
        image: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool = True,
        unit: None | u.UnitBase = None,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        # the matrices spread an image over every wavelength, so they only
        # give the backprojection of an image integrated along wavelength
        result = None
        if integrate:
            result = self._backproject_matrices(image)

        if result is None:
            result = self._backproject_weights(image, integrate=integrate)

        # the backprojection is in energy units; express it in the
        # requested unit, converting to photons if necessary.
        return result.replace(outputs=self._to_unit(result.outputs, unit))

    def _backproject_matrices(
        self,
        image: na.AbstractScalar | na.AbstractFunctionArray,
    ) -> None | na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The backprojection of an image integrated along the wavelength axis,
        in energy units, computed with the sparse matrices of this
        instrument, or :obj:`None` if the instrument or the image cannot be
        expressed as plain arrays, which leaves the backprojection to
        :meth:`_backproject_weights`.

        Any axes of the image beyond those of the channels and the sensor
        are those of independent frames, which are backprojected in the
        same pass over the matrix.

        Parameters
        ----------
        image
            The images captured by this instrument.
        """
        operators = self._operators()
        if operators is None:
            return None

        if isinstance(image, na.AbstractFunctionArray):
            if not np.all(image.inputs.position == self.coordinates_sensor.position):
                raise ValueError(
                    "`image.inputs` and `self.coordinates_sensor` are not equal."
                )
            image = image.outputs

        shape_voxels = operators.shape_channel | operators.shape_scene

        # every other axis of the image is an axis of the frames, unless it
        # is one of those of the scene, which the named arrays are left to
        shape_frames = frames(
            shape=na.shape(image),
            shape_own=operators.shape_images,
            shape_other=shape_voxels,
        )
        if shape_frames is None:
            return None

        values = framed(
            a=image,
            shape=operators.shape_images,
            shape_frames=shape_frames,
            unit=operators.unit_images,
        )
        if values is None:
            return None

        result = operators.backproject(values)
        result *= operators.scale_backproject
        result = result.reshape(result.shape[:~0] + tuple(shape_frames.values()))

        return na.FunctionArray(
            inputs=self.coordinates_scene,
            outputs=na.ScalarArray(
                ndarray=result << operators.unit_scene,
                axes=(*shape_voxels, *shape_frames),
            ),
        )

    def _backproject_weights(
        self,
        image: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        The backprojection of an image, in energy units, computed from
        :attr:`weights_transpose`, one wavelength at a time, as named arrays.

        This is the backprojection of any instrument and any image, and the
        only one of an image which keeps the wavelengths separate.

        Parameters
        ----------
        image
            The images captured by this instrument.
        integrate
            Whether the image is integrated along the wavelength axis.
        """

        # convert the measured electrons back into photons
        if isinstance(image, na.AbstractFunctionArray):
            image = image.replace(outputs=image.outputs / self.quantum_yield)
        else:
            image = image / self.quantum_yield

        result = super().backproject(
            image=image,
            integrate=integrate,
        )

        return result / (self.area_effective * self.timedelta_exposure)


@dataclasses.dataclass
class OptikaInstrument(
    AbstractLinearInstrument,
):
    """
    A CTIS instrument whose forward model is an :mod:`optika`
    :class:`~optika.systems.AbstractLinearSystem`.

    The optika system supplies the distortion, effective area, and vignetting;
    this class adapts its regridding forward model to the
    :class:`AbstractLinearInstrument` interface and adds the transpose
    (:meth:`backproject`) used during inversion. The system may be
    *channel-aware*: its component models can vary along :attr:`axis_channel`
    to represent the different CTIS projections.

    The inverters iterate on a pair of sparse matrices which this instrument
    assembles from :attr:`weights`, with the response of its sensor at each
    wavelength folded into them, so their images are those of :meth:`image`.
    The variance of the noise of those images is taken from one
    variance-to-mean ratio of the sensor for all the wavelengths of a
    spectral line, so a scene of several lines names the axis which
    separates them, :attr:`axis_line`.
    If the ratio varies by more than two parts in a thousand across the
    wavelengths of a line, the instrument does not provide the matrices,
    and the inverters iterate on :meth:`image` and :meth:`backproject`,
    which is slower.
    """

    _tolerance_vmr: ClassVar[float] = 2e-3
    """
    The largest change of the variance-to-mean ratio of the sensor across
    the wavelengths of a spectral line, relative to the middle of its range,
    for which this instrument provides operators.
    """

    system: optika.systems.AbstractLinearSystem = dataclasses.MISSING
    """A :mod:`optika` representation of a linear optical system."""

    coordinates_scene: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    """
    A grid of wavelength and position coordinates on the skyplane
    which will be used to construct the inverted scene.

    Normally the pitch of this grid is chosen to be the average
    plate scale of the instrument.
    """

    channel: str | na.AbstractScalar = dataclasses.MISSING
    """
    Human-readable name of each independent CTIS channel.
    """

    axis_channel: str | tuple[str, ...] = dataclasses.MISSING
    """
    The logical axis or axes of :attr:`system` corresponding to the different
    CTIS channels.
    """

    axis_wavelength: str = dataclasses.MISSING
    """
    The logical axis of :attr:`coordinates_scene` corresponding to changing
    wavelength coordinate.
    """

    axis_scene_xy: tuple[str, str] = dataclasses.MISSING
    """
    The logical axes of :attr:`coordinates_scene` corresponding to changing
    position coordinate.
    """

    axis_line: None | str = None
    """
    The logical axis of :attr:`coordinates_scene` which separates the
    spectral lines of the scene, or :obj:`None` (the default) if the scene is
    one band of wavelengths.

    Each line has its own wavelengths along :attr:`axis_wavelength`, the same
    number for every line, and all the lines are imaged onto the same
    sensors.
    An image which is integrated along wavelength is therefore summed over
    the lines too, and the backprojection of one is spread over the
    wavelengths of every line.
    """

    @property
    def axis_sensor_xy(self) -> tuple[str, str]:
        axis_pixel = self.system.sensor.axis_pixel
        return (axis_pixel.x, axis_pixel.y)

    @property
    def coordinates_sensor(self) -> na.AbstractSpectralPositionalVectorArray:
        return na.SpectralPositionalVectorArray(
            wavelength=self.coordinates_scene.wavelength,
            position=self.system.coordinates_sensor,
        )

    @functools.cached_property
    def weights(self) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        return self.system.weights(
            coordinates=self.coordinates_scene,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_scene_xy,
        )

    @functools.cached_property
    def weights_transpose(
        self,
    ) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        return self.system.weights_transposed(
            weights=self.weights,
            coordinates=self.coordinates_scene,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_scene_xy,
        )

    def _factors(
        self,
        shape_channel: dict[str, int],
        shape_line: dict[str, int],
        shape_scene: dict[str, int],
        shape_sensor: dict[str, int],
        dtype: np.dtype,
    ) -> None | dict[str, np.ndarray | u.UnitBase]:
        """
        The factors around the matrices of the operators of this instrument,
        and the units of the arrays they act on, as the fields of
        :class:`~ctis.instruments._operators.Operators` of the same names,
        or :obj:`None` if this instrument cannot be expressed as those
        factors.

        The response of the sensor is not derived here: the sensor is asked
        for it, by exposing one pixel to one photon a second at each
        wavelength, without noise.
        The electrons this gives at each wavelength are folded into the
        factor of every voxel, ahead of the sum over wavelength, so the
        images are those of :meth:`image`.
        The variance of their noise is the images of each spectral line
        times one variance-to-mean ratio for all the wavelengths of the
        line, the middle of its range, which is why an instrument whose
        ratio varies by more than :attr:`_tolerance_vmr` across a line does
        not provide operators.

        Parameters
        ----------
        shape_channel
            The axes of the channels and their lengths.
        shape_line
            The axis of the spectral lines and its length, if the scene has
            several.
        shape_scene
            The axes of a scene and their lengths.
        shape_sensor
            The two spatial axes of an image and their lengths.
        dtype
            The precision of the factors.
        """
        system = self.system
        sensor = system.sensor
        direction = system.direction

        axis_wavelength = self.axis_wavelength
        axis_pixel = self.axis_sensor_xy
        num_wavelength = shape_scene[axis_wavelength]

        shape_voxels = shape_channel | shape_scene
        shape_pixels = shape_channel | shape_sensor

        # one pixel, since the response of the sensor is that of any of them
        shape_pixel = {ax: 1 for ax in axis_pixel}
        shape_response = shape_channel | shape_line
        shape_response = shape_response | {axis_wavelength: num_wavelength}

        inputs = na.SpectralPositionalVectorArray(
            wavelength=self.coordinates_scene.spectral_positional.wavelength,
            position=na.Cartesian2dVectorArray(0, 0) * u.pix,
        )
        ones = na.ScalarArray(
            ndarray=np.ones((num_wavelength, 1, 1)),
            axes=(axis_wavelength, *axis_pixel),
        )

        # the signal, and the width of its noise, which one photon a second
        # at each wavelength gives
        rate = 1 * u.photon / u.s
        exposed = sensor.expose(
            image=na.FunctionArray(inputs=inputs, outputs=ones * rate),
            direction=direction,
            axis_wavelength=axis_wavelength,
            noise=False,
            integrate=False,
            uncertainty=True,
        ).outputs
        gain = exposed.nominal / rate
        vmr = np.square(exposed.width) / exposed.nominal

        unit_images = na.unit_normalized(exposed.nominal)

        # the photons a second, at each wavelength, which one unit of signal
        # in a pixel is taken for
        absorbed = sensor.photons_absorbed(
            image=na.FunctionArray(
                inputs=inputs,
                outputs=ones[{axis_wavelength: 0}] * unit_images,
            ),
            direction=direction,
            axis_wavelength=axis_wavelength,
            integrate=True,
        ).outputs

        volume = self._volume_scene
        area = 1 * system.weights_unit

        # the unit that `backproject` gives its result in
        unit_scene = u.photon / u.s / system.weights_unit
        unit_scene = unit_scene / na.unit_normalized(volume)

        # the factors that `image` applies to the scene around its
        # resampling, and those that `backproject` applies around its own
        # The sensor spreads an image over the wavelengths of one line, and
        # the backprojection spreads it over those of every line
        scale = (1 * unit_scene) * volume * area * gain
        scale_backproject = absorbed / area / volume
        scale_backproject = scale_backproject / math.prod(shape_line.values())

        scale = aligned(scale, shape=shape_voxels | shape_pixel, unit=unit_images)
        scale_backproject = aligned(
            a=scale_backproject,
            shape=shape_voxels | shape_pixel,
            unit=unit_scene,
        )
        vmr = aligned(vmr, shape=shape_response | shape_pixel, unit=unit_images)
        read_noise = aligned(sensor.read_noise, shape=shape_pixels, unit=unit_images)

        # anything with other axes or units is left to the named arrays
        if any(a is None for a in (scale, scale_backproject, vmr, read_noise)):
            return None

        # one ratio for all the wavelengths of a line, the middle of its
        # range, which is that of each of them to within the tolerance
        axis = len(shape_channel) + len(shape_line)
        middle = (vmr.max(axis, keepdims=True) + vmr.min(axis, keepdims=True)) / 2
        if not np.all(np.abs(vmr - middle) <= self._tolerance_vmr * middle):
            return None

        num_axes = len(shape_voxels)
        scale = scale.reshape(scale.shape[:num_axes])
        scale_backproject = scale_backproject.reshape(
            scale_backproject.shape[:num_axes]
        )
        vmr = middle.reshape(middle.shape[:axis])

        # the response of the sensor is in the factor of every voxel
        gain = np.ones((1,) * axis)

        # the factors of the voxels are given one more axis, which the
        # frames are broadcast along, and every factor the precision of the
        # matrices
        return dict(
            scale=scale[..., np.newaxis].astype(dtype),
            scale_backproject=scale_backproject[..., np.newaxis].astype(dtype),
            gain=gain.astype(dtype),
            vmr=vmr.astype(dtype),
            variance_read=np.square(read_noise).astype(dtype),
            unit_scene=unit_scene,
            unit_images=unit_images,
        )

    def _grids_weights(
        self,
    ) -> tuple[na.AbstractCartesian2dVectorArray, na.AbstractCartesian2dVectorArray]:
        system = self.system

        coordinates = self.coordinates_scene.spectral_positional
        coordinates = coordinates.cell_centers(self.axis_wavelength)

        return (
            system.distortion.distort(coordinates).position,
            system.coordinates_sensor,
        )

    def _weights_input(self) -> na.AbstractScalar:
        system = self.system

        coordinates = self.coordinates_scene.spectral_positional
        coordinates = coordinates.cell_centers(self.axis_wavelength)
        coordinates = coordinates.cell_centers(self.axis_scene_xy)

        # the effective area and the vignetting of every cell are what the
        # transposed weights of optika invert, and its field stop is not
        result = system.area_effective(coordinates.wavelength)
        result = result.to_value(system.weights_unit)

        vignetting = system.vignetting
        if vignetting is not None:
            result = result * vignetting(coordinates)

        return result

    def image(
        self,
        scene: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool = True,
        noise: bool = True,
        uncertainty: bool = False,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        if isinstance(scene, na.AbstractFunctionArray):
            if not np.all(scene.inputs == self.coordinates_scene):
                raise ValueError(
                    "`scene.inputs` and `self.coordinates_scene` are not equal."
                )
        else:
            scene = na.FunctionArray(inputs=self.coordinates_scene, outputs=scene)

        # optika integrates an image over the wavelengths of one line, and
        # adds the read noise of the sensor to it, so the lines of a scene
        # which has several are summed here instead, and the read noise is
        # added once
        summed = integrate and self.axis_line is not None

        # optika applies the effective area, vignetting, and sensor response
        # (including the integration over wavelength and read noise) using the
        # cached regridding weights.
        result = self.system.image_from_weights(
            self.weights,
            scene,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_scene_xy,
            noise=noise,
            uncertainty=uncertainty,
            integrate=integrate and not summed,
        )

        if not summed:
            return result

        axis = self._axis_spectral
        read_noise = self.system.sensor.read_noise

        electrons = result.outputs

        if uncertainty:
            # the variances of the wavelengths of every line add
            variance = np.square(electrons.width).sum(axis)
            width = np.sqrt(variance + np.square(read_noise))
            electrons = electrons.nominal

        electrons = electrons.sum(axis)

        if noise:
            electrons = na.random.normal(loc=electrons, scale=read_noise)

        if uncertainty:
            electrons = na.NormalUncertainScalarArray(nominal=electrons, width=width)

        return na.FunctionArray(
            inputs=self._collapse_wavelength(result.inputs),
            outputs=electrons,
        )

    def backproject(
        self,
        image: na.AbstractScalar | na.AbstractFunctionArray,
        integrate: bool = True,
        unit: None | u.UnitBase = None,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        if isinstance(image, na.AbstractFunctionArray):
            if not np.all(image.inputs.position == self.coordinates_sensor.position):
                raise ValueError(
                    "`image.inputs` and `self.coordinates_sensor` are not equal."
                )
            values = image.outputs
        else:
            values = image

        # rebuild the detector image over the full sensor wavelength grid; optika
        # spreads the integrated readout back over wavelength and inverts the
        # sensor response with the cached transpose weights.
        image = na.FunctionArray(
            inputs=self.coordinates_sensor,
            outputs=values,
        )

        # the energy/photon conversion is handled by optika's backproject.
        result = self.system.backproject_from_weights(
            self.weights_transpose,
            image,
            coordinates=self.coordinates_scene,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_scene_xy,
            integrate=integrate,
            unit=unit,
        )

        # optika spreads an integrated image over the wavelengths of one
        # line, and the backprojection spreads it over those of every line,
        # so that it conserves the total of the image
        axis_line = self.axis_line
        if integrate and axis_line is not None:
            num_line = self.coordinates_scene.wavelength.shape[axis_line]
            result = result.replace(outputs=result.outputs / num_line)

        return result
