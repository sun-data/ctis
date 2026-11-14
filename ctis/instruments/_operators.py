"""
An instrument as a pair of sparse matrices which act on plain arrays.
"""

from typing import Any
import dataclasses
import math
import numpy as np
import numba
import astropy.units as u
import named_arrays as na
import regridding

__all__ = [
    "Operators",
    "aligned",
    "framed",
    "frames",
    "regridder",
    "vertices",
]


def aligned(
    a: Any,
    shape: dict[str, int],
    unit: u.UnitBase,
    dtype: type = np.float64,
) -> None | np.ndarray:
    """
    The values of a named array as a plain array which broadcasts against
    arrays of a given shape.

    The result has the axes of `shape`, in order, each with the length it has
    in `a`, or a length of one if `a` does not vary along it.
    It shares the memory of `a` if `a` is already in `unit` and in `dtype`,
    so it is not to be overwritten.

    It is :obj:`None` if `a` cannot be expressed this way: if it is not an
    array of scalars, if it has an axis which `shape` does not have or one of
    another length, or if it cannot be converted to `unit`.
    The caller is expected to fall back on the named arrays themselves, which
    then handle `a` or raise the error it deserves.

    Parameters
    ----------
    a
        The array, or anything which can be converted to one.
    shape
        The axes of the result and the lengths `a` can have along them.
    unit
        The unit of the result.
    dtype
        The precision of the result.
    """
    a = na.as_named_array(a)

    if not isinstance(a, na.AbstractScalarArray):
        return None

    a = a.explicit

    if any(ax not in shape or n not in (1, shape[ax]) for ax, n in a.shape.items()):
        return None

    # a view of the values, where the unit and the precision allow one
    result = a.ndarray_aligned(tuple(shape))
    if not isinstance(result, u.Quantity):
        result = result << u.dimensionless_unscaled

    try:
        result = result.to_value(unit)
    except u.UnitConversionError:
        return None

    return np.asarray(result, dtype=dtype)


def framed(
    a: Any,
    shape: dict[str, int],
    shape_frames: dict[str, int],
    unit: u.UnitBase,
    dtype: type = np.float64,
) -> None | np.ndarray:
    """
    The values of a named array as a plain, contiguous array of a given
    shape, followed by one axis for the frames.

    It is :obj:`None` if `a` cannot be expressed this way, see
    :func:`aligned`.
    The result can share the memory of `a`, so it is not to be overwritten.

    Parameters
    ----------
    a
        The array, or anything which can be converted to one.
    shape
        The axes which come first in the result and their lengths.
        If `a` does not vary along one of them, it is broadcast along it.
    shape_frames
        The axes of the frames and their lengths, which are merged into the
        last axis of the result.
    unit
        The unit of the result.
    dtype
        The precision of the result.
    """
    result = aligned(a, shape=shape | shape_frames, unit=unit, dtype=dtype)
    if result is None:
        return None

    shape_result = tuple(shape.values())
    result = np.broadcast_to(result, shape_result + tuple(shape_frames.values()))
    result = np.ascontiguousarray(result, dtype=dtype)

    return result.reshape(shape_result + (math.prod(shape_frames.values()),))


def frames(
    shape: dict[str, int],
    shape_own: dict[str, int],
    shape_other: dict[str, int],
) -> None | dict[str, int]:
    """
    The axes of the frames of a scene or of images, and their lengths: every
    axis of the array beyond its own.

    It is :obj:`None` if one of them is an axis of the other kind of array,
    since a scene with an axis of the images, or images with an axis of a
    scene, are not a stack of frames.
    The caller is expected to fall back on the named arrays themselves.

    Parameters
    ----------
    shape
        The axes of the array and their lengths.
    shape_own
        The axes that one frame of the array has: those of a scene, or those
        of the images.
    shape_other
        The axes of the other kind of array, which no axis of the frames can
        be named after.
    """
    result = {ax: num for ax, num in shape.items() if ax not in shape_own}

    if not result.keys().isdisjoint(shape_other):
        return None

    return result


def vertices(
    coordinates: na.AbstractVectorArray,
    shape: dict[str, int],
    unit: u.UnitBase,
) -> None | tuple[np.ndarray, ...]:
    """
    The components of the vertices of a grid as plain arrays of a given
    shape, or :obj:`None` if any of them cannot be expressed that way,
    see :func:`aligned`.

    Parameters
    ----------
    coordinates
        The vertices of the grid.
    shape
        The axes of the result and their lengths.
    unit
        The unit of the result.
    """
    result = []
    for component in coordinates.components.values():
        component = aligned(component, shape=shape, unit=unit)
        if component is None:
            return None
        result.append(np.broadcast_to(component, tuple(shape.values())))
    return tuple(result)


def regridder(
    weights: tuple[na.AbstractScalarArray, dict[str, int], dict[str, int]],
    shape_input: dict[str, int],
    shape_output: dict[str, int],
    dtype: type = np.float64,
) -> regridding.Regridder:
    """
    Assemble weights computed by :func:`named_arrays.regridding.weights` into
    one sparse matrix, which acts on plain arrays.

    Parameters
    ----------
    weights
        The weights, with the shapes of the grids they were computed for.
        The axes they are an array over are the orthogonal axes, and the
        other axes of the grids are the resampled axes.
    shape_input
        The shape of the arrays the matrix is applied to: the orthogonal
        axes, followed by the resampled axes of the input grid in the order
        the grid has them.
        An orthogonal axis can have a length of one, which applies every
        element of the weights along it to the same input.
    shape_output
        The shape of the arrays the matrix returns: the orthogonal axes, in
        the same order as in `shape_input`, followed by the resampled axes of
        the output grid.
        An orthogonal axis can have a length of one, which sums what the
        elements of the weights along it give.
    dtype
        The type of the entries of the matrix, :obj:`numpy.float64` or
        :obj:`numpy.float32`.
    """
    array, shape_grid_input, shape_grid_output = weights

    axis_input = tuple(ax for ax in shape_grid_input if ax not in array.shape)
    axis_output = tuple(ax for ax in shape_grid_output if ax not in array.shape)

    num_orthogonal = len(shape_input) - len(axis_input)
    axis_orthogonal = tuple(shape_input)[:num_orthogonal]

    if tuple(shape_input) != axis_orthogonal + axis_input:
        raise ValueError(
            f"the input should have the resampled axes of its grid, "
            f"{axis_input}, last, but its axes are {tuple(shape_input)}"
        )
    if tuple(shape_output) != axis_orthogonal + axis_output:
        raise ValueError(
            f"the output should have the axes {axis_orthogonal + axis_output}, "
            f"but its axes are {tuple(shape_output)}"
        )

    array = array.ndarray_aligned(axis_orthogonal)

    result = regridding.Regridder.from_weights(
        weights=(
            array,
            array.shape + tuple(shape_grid_input[ax] for ax in axis_input),
            array.shape + tuple(shape_grid_output[ax] for ax in axis_output),
        ),
        axis_input=tuple(range(num_orthogonal, len(shape_input))),
        axis_output=tuple(range(num_orthogonal, len(shape_output))),
        shape_input=tuple(shape_input.values()),
        shape_output=tuple(shape_output.values()),
        dtype=dtype,
    )

    # weights with a dimensionless unit, as those weighted by a vignetting
    # model have, are given none, so that the products of the matrix are
    # the plain arrays which the compiled code takes
    if result.unit is not None:
        result.data *= u.Quantity(1, result.unit).to_value(u.dimensionless_unscaled)
        result.unit = None

    return result


@numba.njit(cache=True, parallel=True)
def _multiply(
    scene: np.ndarray,
    scale: np.ndarray,
    result: np.ndarray,
) -> None:
    """
    Multiply the scene of each frame by a factor for each channel and each
    voxel.

    This is one pass of :func:`numpy.multiply`, shared between the threads.

    Parameters
    ----------
    scene
        The scenes, an array with an axis for the wavelengths, two for the
        positions and one for the frames.
    scale
        The factors, an array with an axis for the channels followed by
        those of a scene, which need not be contiguous: a factor which
        does not vary along an axis can be broadcast along it.
    result
        An array with an axis for the channels followed by those of `scene`,
        which is overwritten with the products.
    """
    num_channel, num_wavelength, num_x, num_y, num_frame = result.shape

    for k in numba.prange(num_channel * num_wavelength):
        c = k // num_wavelength
        w = k % num_wavelength
        for x in range(num_x):
            for y in range(num_y):
                factor = scale[c, w, x, y]
                for f in range(num_frame):
                    result[c, w, x, y, f] = scene[w, x, y, f] * factor


@dataclasses.dataclass(eq=False)
class Operators:
    """
    An instrument as a pair of sparse matrices, one for its forward model and
    one for its backprojection, which act on plain arrays with neither named
    axes nor units.

    These reproduce the :meth:`~ctis.instruments.AbstractInstrument.image`
    and :meth:`~ctis.instruments.AbstractInstrument.backproject` methods of
    the instrument they came from, without the cost of the named arrays, and
    are what the inverters iterate on if the instrument provides them.

    The matrix of the backprojection is derived from that of the forward
    model the first time it is needed, so that an instrument which is only
    used to image scenes holds one matrix and not two.

    A scene is an array of shape :attr:`shape_scene` in units of
    :attr:`unit_scene`, and the images are an array of shape
    :attr:`shape_images` in units of :attr:`unit_images`.

    Both are followed by one more axis for the frames, the independent sets
    of images of one instrument which are computed together.
    The matrices are applied along that axis in a single pass, in which each
    frame gets the result it would have alone, bit for bit.
    The frames come last because that is how the matrices store them, so
    that no array is rearranged on its way in or out.

    The matrices, the factors around them and the arrays they act on are all
    in one precision, :attr:`dtype`.
    """

    forward: regridding.Regridder
    """
    The matrix which takes the photons leaving every voxel of the scene
    toward each channel to the photons landing in every pixel of that
    channel, summed over wavelength.
    """

    vertices_scene: tuple[np.ndarray, ...]
    """
    The components of the vertices of the cells of the scene, where each
    channel puts them on the sensor at each wavelength, which
    :attr:`backward` is derived from.
    """

    vertices_sensor: tuple[np.ndarray, ...]
    """
    The components of the vertices of the pixels of the sensor, for each
    channel and each wavelength, which :attr:`backward` is derived from.
    """

    scale: np.ndarray
    """
    The photons leaving each voxel toward each channel for every unit of the
    scene there, which broadcasts against the channels, the scene and the
    frames.
    """

    scale_backproject: np.ndarray
    """
    The factor, for each channel and each voxel, between what :meth:`backproject`
    returns and the backprojection of the instrument in units of
    :attr:`unit_scene`, which broadcasts against the channels, the scene and
    the frames.
    """

    gain: np.ndarray
    """
    The signal measured for each unit of what :attr:`forward` brings to a
    pixel, which broadcasts against the images and the frames.
    """

    vmr: np.ndarray
    """
    The variance of the shot noise of a pixel for each unit of its signal,
    the variance-to-mean ratio, in units of :attr:`unit_images`, which
    broadcasts against the images and the frames.
    """

    variance_read: np.ndarray
    """
    The variance of the read noise, in units of :attr:`unit_images` squared,
    which broadcasts against the images and the frames.
    """

    shape_channel: dict[str, int]
    """The axes of the channels of the instrument and their lengths."""

    shape_scene: dict[str, int]
    """
    The axes of a scene and their lengths: the wavelength axis, followed by
    the two spatial axes.
    """

    shape_sensor: dict[str, int]
    """The two spatial axes of an image and their lengths."""

    unit_scene: u.UnitBase
    """The unit of a scene."""

    unit_images: u.UnitBase
    """The unit of the images."""

    weights_scene: None | np.ndarray = None
    """
    The weight that :attr:`forward` gives every voxel of the scene, for each
    channel, beyond the fraction of the voxel which lands in each pixel,
    such as the effective area of the instrument there, or :obj:`None` if it
    gives none.

    The matrix of the backprojection inverts this weight, as
    :func:`regridding.transpose_weights_conservative` does.
    """

    _scale: np.ndarray = dataclasses.field(init=False, repr=False)
    """
    :attr:`scale` as :meth:`image` applies it: a view with one axis for the
    channels, as many as :attr:`forward` takes, followed by those of a scene.
    """

    _backward: None | regridding.Regridder = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )
    """
    The matrix of the backprojection, once :attr:`backward` has derived it.
    """

    def __post_init__(self) -> None:

        num_axis_channel = len(self.shape_channel)
        shape_channel = self.forward.shape_input[:num_axis_channel]
        shape_scene = self.forward.shape_input[num_axis_channel:]

        # the axes of the channels are merged before the factor is broadcast
        # along the scene, so that merging them copies as little as it can
        scale = self.scale[..., 0]
        shape_factor = scale.shape[num_axis_channel:]
        scale = np.broadcast_to(scale, shape_channel + shape_factor)
        scale = scale.reshape((math.prod(shape_channel),) + shape_factor)

        self._scale = np.broadcast_to(scale, scale.shape[:1] + shape_scene)

    @property
    def backward(self) -> regridding.Regridder:
        """
        The matrix which spreads the signal in every pixel of every channel
        over the voxels which could have contributed to it, for each channel.

        It is the transpose of :attr:`forward`, normalized to conserve flux,
        and it is derived the first time it is asked for.
        """
        if self._backward is None:
            self._backward = self.forward.transpose_conservative(
                coordinates_input=self.vertices_scene,
                coordinates_output=self.vertices_sensor,
                weights_input=self.weights_scene,
            )
        return self._backward

    @property
    def shape_images(self) -> dict[str, int]:
        """
        The axes of the images and their lengths: the axes of the channels,
        followed by the two spatial axes.
        """
        return self.shape_channel | self.shape_sensor

    @property
    def num_channel(self) -> int:
        """The total number of channels."""
        return math.prod(self.shape_channel.values())

    @property
    def dtype(self) -> np.dtype:
        """
        The precision of the matrices, which the arrays they act on are
        expected in, and their results are given in.
        """
        return np.dtype(self.forward.data.dtype)

    def photons(
        self,
        scene: np.ndarray,
        buffer: None | np.ndarray = None,
    ) -> np.ndarray:
        """
        The photons expected in every pixel of the images of the scene of
        each frame, summed over wavelength, without noise.

        The photons are summed over wavelength before those of a pixel are
        kept from being negative, so that the forward model is linear up to
        the point where the photons are counted.

        The result is a new array of shape :attr:`shape_images` followed by
        an axis for the frames, which the caller is free to overwrite.

        Parameters
        ----------
        scene
            The scenes, of shape :attr:`shape_scene` followed by an axis for
            the frames.
        buffer
            An array to overwrite with the photons leaving every voxel
            toward each channel: a contiguous array of the shape that
            :attr:`forward` takes followed by an axis for the frames, in the
            precision of the matrices.
            An iteration gives every call the same one, which is faster than
            allocating it again.
            If :obj:`None`, a new array is allocated.
        """
        shape_scene = tuple(self.shape_scene.values())
        if scene.shape[:~0] != shape_scene:
            raise ValueError(
                f"the scenes should have a shape of {shape_scene} followed by "
                f"an axis for the frames, but their shape is {scene.shape}"
            )

        num_frame = scene.shape[~0]

        # the photons which leave every voxel, written straight into an
        # array of the shape the matrix takes, which can have a scene for
        # each channel, rather than broadcast to it by the matrix afterward
        shape = self.forward.shape_input + (num_frame,)
        if buffer is None:
            buffer = np.empty(shape, dtype=self.dtype)
        elif (
            buffer.shape != shape
            or buffer.dtype != self.dtype
            or not buffer.flags.c_contiguous
        ):
            raise ValueError(
                f"the buffer should be a contiguous array of shape {shape} "
                f"and type {self.dtype}, but it is one of shape {buffer.shape} "
                f"and type {buffer.dtype}"
            )
        _multiply(
            scene=scene,
            scale=self._scale,
            result=buffer.reshape(self._scale.shape + (num_frame,)),
        )

        photons = self._apply(self.forward, buffer)
        photons = photons.reshape(tuple(self.shape_images.values()) + (num_frame,))
        np.maximum(photons, 0, out=photons)

        return photons

    def image(
        self,
        scene: np.ndarray,
        buffer: None | np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The images expected of the scene of each frame, without noise, and
        the variance of their noise.

        Parameters
        ----------
        scene
            The scenes, of shape :attr:`shape_scene` followed by an axis for
            the frames.
        buffer
            An array to overwrite with the photons leaving every voxel, see
            :meth:`photons`.
        """
        images = self.photons(scene, buffer=buffer)
        images *= self.gain

        # the variance of the shot noise, which grows with the signal
        variance = images * self.vmr
        variance += self.variance_read

        return images, variance

    def backproject(self, images: np.ndarray) -> np.ndarray:
        """
        Spread the images of each frame evenly over the voxels which could
        have contributed to them, giving a scene for each channel.

        The result is the backprojection of the instrument divided, for each
        channel and each voxel, by :attr:`scale_backproject`.
        The ratio of two backprojections, which is all that an algorithm such
        as MART needs, is the same either way.

        Parameters
        ----------
        images
            The images, of shape :attr:`shape_images` followed by an axis for
            the frames.
        """
        # images of another size cannot be given the shape the matrix takes,
        # so they raise an error here
        num_frame = images.shape[~0]
        images = images.reshape(self.backward.shape_input + (num_frame,))
        return self._apply(self.backward, images)

    @staticmethod
    def _apply(matrix: regridding.Regridder, values: np.ndarray) -> np.ndarray:
        """
        Apply a matrix to values with the frames along their last axis, and
        return the result with the frames along its last axis.

        The matrix takes the frames along a leading axis, and stores them
        along the last, so a view of the values goes in, and a view of the
        result comes out, without either being copied.

        Parameters
        ----------
        matrix
            The matrix to apply.
        values
            The values, of the shape the matrix takes followed by an axis
            for the frames.
        """
        result = matrix(np.moveaxis(values, ~0, 0))
        return np.moveaxis(result, 0, ~0)
