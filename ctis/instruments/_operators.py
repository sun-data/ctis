"""
An instrument as a pair of sparse matrices which act on plain arrays.
"""

from typing import Any
import dataclasses
import math
import numpy as np
import astropy.units as u
import named_arrays as na
import regridding

__all__ = [
    "Operators",
    "aligned",
    "regridder",
    "vertices",
]


def aligned(
    a: Any,
    shape: dict[str, int],
    unit: u.UnitBase,
) -> None | np.ndarray:
    """
    The values of a named array as a plain array which broadcasts against
    arrays of a given shape.

    The result has the axes of `shape`, in order, each with the length it has
    in `a`, or a length of one if `a` does not vary along it.

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
    """
    a = na.as_named_array(a)

    if not isinstance(a, na.AbstractScalarArray):
        return None

    a = a.explicit

    if any(ax not in shape or n not in (1, shape[ax]) for ax, n in a.shape.items()):
        return None

    try:
        result = u.Quantity(a.ndarray_aligned(tuple(shape))).to_value(unit)
    except u.UnitConversionError:
        return None

    return np.asarray(result, dtype=float)


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

    return regridding.Regridder.from_weights(
        weights=(
            array,
            array.shape + tuple(shape_grid_input[ax] for ax in axis_input),
            array.shape + tuple(shape_grid_output[ax] for ax in axis_output),
        ),
        axis_input=tuple(range(num_orthogonal, len(shape_input))),
        axis_output=tuple(range(num_orthogonal, len(shape_output))),
        shape_input=tuple(shape_input.values()),
        shape_output=tuple(shape_output.values()),
    )


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

    A scene is an array of shape :attr:`shape_scene` in units of
    :attr:`unit_scene`, and the images are an array of shape
    :attr:`shape_images` in units of :attr:`unit_images`.
    """

    forward: regridding.Regridder
    """
    The matrix which takes the photons leaving every voxel of the scene
    toward each channel to the photons landing in every pixel of that
    channel, summed over wavelength.
    """

    backward: regridding.Regridder
    """
    The matrix which spreads the signal in every pixel of every channel over
    the voxels which could have contributed to it, for each channel.
    """

    scale: np.ndarray
    """
    The photons leaving each voxel toward each channel for every unit of the
    scene there, which broadcasts against the channels and the scene.
    """

    scale_backproject: np.ndarray
    """
    The factor, for each channel and each voxel, between what :meth:`backproject`
    returns and the backprojection of the instrument in units of
    :attr:`unit_scene`.
    """

    gain: np.ndarray
    """
    The signal measured for each photon landing in a pixel, which broadcasts
    against the images.
    """

    variance_read: np.ndarray
    """
    The variance of the read noise, in units of :attr:`unit_images` squared,
    which broadcasts against the images.
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

    def image(self, scene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        The images expected of a scene, without noise, and the variance of
        their noise.

        Parameters
        ----------
        scene
            A scene, of shape :attr:`shape_scene`.
        """
        # the photons which leave every voxel, written straight into an
        # array of the shape the matrix takes, which can have a scene for
        # each channel, rather than broadcast to it by the matrix afterward
        photons = np.empty(self.forward.shape_input)
        np.multiply(scene, self.scale, out=photons)

        photons = self.forward(photons)
        photons = photons.reshape(tuple(self.shape_images.values()))
        np.maximum(photons, 0, out=photons)

        images = photons
        images *= self.gain

        # the variance of the shot noise of the photons, in units of the images
        variance = images * self.gain
        variance += self.variance_read

        return images, variance

    def backproject(self, images: np.ndarray) -> np.ndarray:
        """
        Spread images evenly over the voxels which could have contributed to
        them, giving a scene for each channel.

        The result is the backprojection of the instrument divided, for each
        channel and each voxel, by :attr:`scale_backproject`.
        The ratio of two backprojections, which is all that an algorithm such
        as MART needs, is the same either way.

        Parameters
        ----------
        images
            The images, of shape :attr:`shape_images`.
        """
        return self.backward(images.reshape(self.backward.shape_input))
