from typing import Any, Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import regridding
from ctis.instruments import _operators

shape = dict(x=3, y=2)


def test_aligned() -> None:
    values = np.arange(6.0).reshape(2, 3)
    a = na.ScalarArray(values, axes=("y", "x")) * u.m

    result = _operators.aligned(a, shape=shape, unit=u.cm)

    assert isinstance(result, np.ndarray)
    assert not isinstance(result, u.Quantity)
    assert np.array_equal(result, 100 * values.T)


@pytest.mark.parametrize(
    argnames="a,expected",
    argvalues=[
        (2 * u.m, np.full((1, 1), 2.0)),
        (na.ScalarArray(np.arange(2), axes="y") * u.m, np.arange(2.0)[np.newaxis]),
        (na.linspace(0, 1, axis="x", num=3) * u.m, np.linspace(0, 1, 3)[:, np.newaxis]),
        (na.ScalarArray(np.ones(1), axes="x") * u.m, np.ones((1, 1))),
    ],
)
def test_aligned_broadcast(a: Any, expected: np.ndarray) -> None:
    """An array which does not vary along an axis has a length of one there."""
    result = _operators.aligned(a, shape=shape, unit=u.m)

    assert result.shape == expected.shape
    assert result.dtype == float
    assert np.array_equal(result, expected)


def test_aligned_dimensionless() -> None:
    result = _operators.aligned(3, shape=shape, unit=u.dimensionless_unscaled)

    assert np.array_equal(result, np.full((1, 1), 3.0))


@pytest.mark.parametrize(
    argnames="a,reason",
    argvalues=[
        (
            na.ScalarArray(np.ones(4), axes="z") * u.m,
            "has the axis 'z', which is not one of ('x', 'y')",
        ),
        (
            na.ScalarArray(np.ones(4), axes="x") * u.m,
            "has 4 elements along 'x', and not 3",
        ),
        (
            na.ScalarArray(np.ones(3), axes="x") * u.s,
            "is in units of `s`, which cannot be converted to `m`",
        ),
        (
            na.ScalarArray(np.ones(3), axes="x"),
            "is dimensionless, which cannot be converted to `m`",
        ),
        (
            na.UniformUncertainScalarArray(1 * u.m, width=1 * u.m, num_distribution=3),
            "is a `UniformUncertainScalarArray`, and not an array of scalars",
        ),
        (
            na.Cartesian2dVectorArray(1, 2) * u.m,
            "is a `Cartesian2dVectorArray`, and not an array of scalars",
        ),
    ],
    ids=[
        "another axis",
        "another length",
        "another unit",
        "no unit",
        "uncertain",
        "vector",
    ],
)
def test_aligned_unsupported(a: Any, reason: str) -> None:
    """
    An array which cannot be expressed as a plain array is left alone,
    and :func:`ctis.instruments._operators.unaligned` says why.
    """
    assert _operators.aligned(a, shape=shape, unit=u.m) is None
    assert _operators.unaligned(a, shape=shape, unit=u.m) == reason


def test_unaligned() -> None:
    """
    An array which can be expressed as a plain array has no reason that it
    cannot, and one with a unit is not a dimensionless number.
    """
    a = na.ScalarArray(np.ones(3), axes="x") * u.m
    unit = u.dimensionless_unscaled

    assert _operators.unaligned(a, shape=shape, unit=u.cm) is None
    assert _operators.unaligned(3, shape=shape, unit=unit) is None
    assert _operators.unaligned(a, shape=shape, unit=unit) == (
        "is in units of `m`, which cannot be converted to a dimensionless number"
    )


def test_vertices() -> None:
    coordinates = na.Cartesian2dVectorArray(
        x=na.linspace(0, 1, axis="x", num=3) * u.m,
        y=na.linspace(2, 3, axis="y", num=2) * u.m,
    )

    result = _operators.vertices(coordinates, shape=shape, unit=u.cm)

    x, y = result
    x_expected = np.linspace(0, 100, 3)[:, np.newaxis]
    y_expected = np.linspace(200, 300, 2)
    assert x.shape == (3, 2)
    assert y.shape == (3, 2)
    assert np.array_equal(x, np.broadcast_to(x_expected, (3, 2)))
    assert np.array_equal(y, np.broadcast_to(y_expected, (3, 2)))


def test_vertices_unsupported() -> None:
    coordinates = na.Cartesian2dVectorArray(
        x=na.linspace(0, 1, axis="x", num=3) * u.m,
        y=na.linspace(2, 3, axis="y", num=2) * u.s,
    )

    result = _operators.vertices(coordinates, shape=shape, unit=u.m)

    assert result == (
        "their component 'y' is in units of `s`, which cannot be converted to `m`"
    )


@pytest.mark.parametrize(
    argnames="multiply",
    argvalues=[_operators._multiply, _operators._multiply.py_func],
    ids=["compiled", "python"],
)
@pytest.mark.parametrize(
    argnames="shape_scale",
    argvalues=[(1, 4, 1, 1), (3, 4, 1, 1), (3, 4, 5, 6)],
    ids=["for each wavelength", "for each channel", "for each voxel"],
)
def test__multiply(multiply: Callable, shape_scale: tuple[int, ...]) -> None:
    """
    The kernel multiplies as numpy does, with a factor which is broadcast
    along the axes it does not vary along.
    """
    rng = np.random.default_rng(0)
    scene = rng.uniform(1, 2, size=(4, 5, 6, 2))
    scale = rng.uniform(1, 2, size=shape_scale)
    scale = np.broadcast_to(scale, shape_scale[:1] + (4, 5, 6))
    result = np.empty(scale.shape + (2,))

    multiply(scene, scale, result)

    assert np.array_equal(result, scene * scale[..., np.newaxis])


def _weights(
    weights_input: None | na.AbstractScalar = None,
) -> tuple[na.AbstractScalarArray, dict[str, int], dict[str, int]]:
    """
    Conservative weights from a grid of 4 by 5 cells, which is shifted by a
    different amount in each of 3 channels, onto a grid of 6 by 4 cells.

    Parameters
    ----------
    weights_input
        The weight of each cell of the input grid, if it has one.
    """
    shift = na.linspace(0, 0.5, axis="channel", num=3)
    coordinates_input = na.Cartesian2dVectorArray(
        x=na.linspace(0, 1, axis="x", num=5) + shift,
        y=na.linspace(0, 1, axis="y", num=6),
    )
    coordinates_output = na.Cartesian2dVectorArray(
        x=na.linspace(0, 1.5, axis="u", num=7),
        y=na.linspace(0, 1, axis="v", num=5),
    )
    return na.regridding.weights(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        axis_input=("x", "y"),
        axis_output=("u", "v"),
        weights_input=weights_input,
        method="conservative",
    )


def _matrix(
    weights: tuple[na.AbstractScalarArray, dict[str, int], dict[str, int]],
    shape_input: dict[str, int],
    shape_output: dict[str, int],
    shape_weights: dict[str, int],
    blocks: list[dict[str, slice]],
) -> regridding.Regridder:
    """
    Assemble weights into a matrix with :class:`regridding.RegridderBuilder`,
    from the pieces that :func:`ctis.instruments._operators.piece` makes of
    the blocks of the weights.

    Parameters
    ----------
    weights
        All of the weights, which each block is a slice of.
    shape_input
        The shape of the arrays the matrix is applied to.
    shape_output
        The shape of the arrays the matrix returns.
    shape_weights
        The orthogonal axes and the length of the weights along each.
    blocks
        The elements of each block.
    """
    array, shape_grid_input, shape_grid_output = weights
    num_orthogonal = len(shape_weights)
    builder = regridding.RegridderBuilder(
        shape_input=tuple(shape_input.values()),
        shape_output=tuple(shape_output.values()),
        axis_input=tuple(range(num_orthogonal, len(shape_input))),
        axis_output=tuple(range(num_orthogonal, len(shape_output))),
        shape_weights=tuple(shape_weights.values()),
    )
    kwargs = dict(
        shape_input=shape_input,
        shape_output=shape_output,
        shape_weights=shape_weights,
    )
    for place in (builder.count, builder.place):
        for index in blocks:
            block = (array[index], shape_grid_input, shape_grid_output)
            place(*_operators.piece(block, index=index, **kwargs))
    result, _ = builder.build()
    return result


@pytest.mark.parametrize(
    argnames="shape_input,shape_output",
    argvalues=[
        (dict(channel=3, x=4, y=5), dict(channel=3, u=6, v=4)),
        (dict(channel=1, x=4, y=5), dict(channel=3, u=6, v=4)),
        (dict(channel=3, x=4, y=5), dict(channel=1, u=6, v=4)),
    ],
    ids=["each", "shared", "summed"],
)
@pytest.mark.parametrize(
    argnames="blocks",
    argvalues=[
        [dict()],
        [dict(channel=slice(c, c + 1)) for c in range(3)],
        [dict(channel=slice(0, 2)), dict(channel=slice(2, 3))],
    ],
    ids=["one block", "a block for each channel", "blocks of two and one"],
)
def test_piece(
    shape_input: dict[str, int],
    shape_output: dict[str, int],
    blocks: list[dict[str, slice]],
) -> None:
    """
    The matrix assembled from the blocks of the weights gives what the
    weights give, with an input shared by the channels if it has only one,
    and summed over the channels if the output has only one.
    """
    weights = _weights()

    rng = np.random.default_rng(0)
    values = rng.uniform(size=tuple(shape_input.values()))

    result = _matrix(
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
        shape_weights=dict(channel=3),
        blocks=blocks,
    )(values)

    expected = na.regridding.regrid_from_weights(
        *weights,
        values_input=na.ScalarArray(values, axes=tuple(shape_input)),
    )
    expected = expected.ndarray_aligned(tuple(shape_output))
    if shape_input["channel"] == 1:
        expected = np.broadcast_to(expected, (3, 6, 4))
    if shape_output["channel"] == 1:
        expected = expected.sum(axis=0, keepdims=True)

    assert result.shape == tuple(shape_output.values())
    assert result.sum() > 0
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_piece_start() -> None:
    """
    A block starts at the start of its slice along each orthogonal axis
    that it does not hold whole, and at zero along the others.
    """
    weights = _weights()
    array, shape_grid_input, shape_grid_output = weights
    index = dict(channel=slice(1, 3))

    (result, shape_piece_input, shape_piece_output), start = _operators.piece(
        weights=(array[index], shape_grid_input, shape_grid_output),
        shape_input=dict(channel=3, x=4, y=5),
        shape_output=dict(channel=3, u=6, v=4),
        shape_weights=dict(channel=3),
        index=index,
    )

    assert start == (1,)
    assert result.shape == (2,)
    assert shape_piece_input == (2, 4, 5)
    assert shape_piece_output == (2, 6, 4)


@pytest.mark.parametrize(
    argnames="shape_input,shape_output",
    argvalues=[
        (dict(x=4, y=5, channel=3), dict(channel=3, u=6, v=4)),
        (dict(channel=3, y=5, x=4), dict(channel=3, u=6, v=4)),
        (dict(channel=3, x=4, y=5), dict(u=6, v=4, channel=3)),
        (dict(channel=3, x=4, y=5), dict(channel=3, v=4, u=6)),
    ],
)
def test_piece_axes(
    shape_input: dict[str, int],
    shape_output: dict[str, int],
) -> None:
    """The orthogonal axes come first, and the resampled axes in grid order."""
    with pytest.raises(ValueError, match="axes"):
        _operators.piece(
            weights=_weights(),
            shape_input=shape_input,
            shape_output=shape_output,
            shape_weights=dict(channel=3),
            index=dict(),
        )


@pytest.mark.parametrize(
    argnames="half",
    argvalues=[50 * u.percent, 0.5 * u.dimensionless_unscaled],
    ids=["percent", "dimensionless"],
)
def test_dimensionless(half: u.Quantity) -> None:
    """
    A matrix assembled from weights with a dimensionless unit is given none,
    and scaled by the unit, so that its products are plain arrays.
    """
    kwargs = dict(
        shape_input=dict(channel=3, x=4, y=5),
        shape_output=dict(channel=3, u=6, v=4),
        shape_weights=dict(channel=3),
        blocks=[dict()],
    )

    weights_input = na.ScalarArray(np.ones((4, 5)), axes=("x", "y")) * half

    matrix = _matrix(weights=_weights(weights_input), **kwargs)
    unweighted = _matrix(weights=_weights(), **kwargs)

    assert matrix.unit is not None
    assert _operators.dimensionless(matrix) is matrix

    rng = np.random.default_rng(0)
    values = rng.uniform(size=(3, 4, 5))

    result = matrix(values)

    assert matrix.unit is None
    assert not isinstance(result, u.Quantity)
    assert result.sum() > 0
    assert np.allclose(result, unweighted(values) / 2, rtol=1e-12, atol=1e-12)


def test_dimensionless_none() -> None:
    """A matrix without a unit is given back as it is."""
    matrix = _matrix(
        weights=_weights(),
        shape_input=dict(channel=3, x=4, y=5),
        shape_output=dict(channel=3, u=6, v=4),
        shape_weights=dict(channel=3),
        blocks=[dict()],
    )
    data = matrix.data.copy()

    assert matrix.unit is None
    assert _operators.dimensionless(matrix) is matrix
    assert np.array_equal(matrix.data, data)


def test_dimensionless_unsupported() -> None:
    """
    A matrix assembled from weights in a unit which is not dimensionless,
    which a matrix that acts on plain arrays cannot carry, raises, since its
    caller is expected to have told so from the weights first.
    """
    weights_input = na.ScalarArray(np.ones((4, 5)), axes=("x", "y")) * u.cm**2

    matrix = _matrix(
        weights=_weights(weights_input),
        shape_input=dict(channel=3, x=4, y=5),
        shape_output=dict(channel=3, u=6, v=4),
        shape_weights=dict(channel=3),
        blocks=[dict()],
    )

    with pytest.raises(u.UnitConversionError):
        _operators.dimensionless(matrix)


@pytest.mark.parametrize(
    argnames="weight,unit",
    argvalues=[
        (None, None),
        (2 * u.cm**2, u.cm**2),
        (50 * u.percent, u.percent),
    ],
    ids=["none", "area", "percent"],
)
def test_unit_weights(
    weight: None | u.Quantity,
    unit: None | u.UnitBase,
) -> None:
    """
    The unit of the values of weights is that of the weight of their cells,
    or none if the cells have no weight.
    """
    weights_input = None
    if weight is not None:
        weights_input = na.ScalarArray(np.ones((4, 5)), axes=("x", "y")) * weight

    assert _operators.unit_weights(_weights(weights_input)) == unit


def test_unit_weights_empty() -> None:
    """Weights without any element have no unit to give."""
    weights_input = na.ScalarArray(np.ones((4, 5)), axes=("x", "y")) * u.cm**2
    array, shape_grid_input, shape_grid_output = _weights(weights_input)
    empty = array[dict(channel=slice(0, 0))]

    assert empty.shape == dict(channel=0)
    assert _operators.unit_weights((empty, shape_grid_input, shape_grid_output)) is None


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_framed(dtype: type) -> None:
    """
    The axes of the frames are merged into one last axis, in the order they
    are given in, and an array which does not vary along an axis is
    broadcast along it.
    """
    shape_frames = dict(time=4, repeat=2)

    rng = np.random.default_rng(0)
    values = rng.uniform(size=(2, 4, 3))
    a = na.ScalarArray(values, axes=("repeat", "time", "x")) * u.m

    result = _operators.framed(
        a=a,
        shape=shape,
        shape_frames=shape_frames,
        unit=u.cm,
        dtype=dtype,
    )

    expected = 100 * values.transpose(2, 1, 0)[:, np.newaxis]
    expected = np.broadcast_to(expected, (3, 2, 4, 2)).reshape(3, 2, 8)

    assert result.shape == (3, 2, 8)
    assert result.dtype == dtype
    assert result.flags.c_contiguous
    assert np.array_equal(result, expected.astype(dtype))


def test_framed_one_frame() -> None:
    """An array without frames has one, along a last axis of length one."""
    values = np.arange(6.0).reshape(3, 2)
    a = na.ScalarArray(values, axes=("x", "y")) * u.m

    result = _operators.framed(a, shape=shape, shape_frames=dict(), unit=u.m)

    assert result.shape == (3, 2, 1)
    assert np.array_equal(result[..., 0], values)


def test_framed_unsupported() -> None:
    """An array which cannot be expressed as a plain array is left alone."""
    a = na.ScalarArray(np.ones(3), axes="x") * u.s

    assert _operators.framed(a, shape=shape, shape_frames=dict(), unit=u.m) is None


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_framed_view(dtype: type) -> None:
    """
    Values which are already in the unit and the precision asked for, with
    their axes in order, are not copied, and values in single precision are
    not converted to double precision on the way.
    """
    values = np.arange(24, dtype=dtype).reshape(3, 2, 4)
    a = na.ScalarArray(values << u.m, axes=("x", "y", "time"))

    result = _operators.framed(
        a=a,
        shape=shape,
        shape_frames=dict(time=4),
        unit=u.m,
        dtype=dtype,
    )

    assert result.dtype == dtype
    assert np.array_equal(result, values)
    assert np.shares_memory(result, values)

    # another unit or another precision needs a copy
    for kwargs in (dict(unit=u.cm, dtype=dtype), dict(unit=u.m, dtype=np.float16)):
        copy = _operators.framed(a=a, shape=shape, shape_frames=dict(time=4), **kwargs)
        assert not np.shares_memory(copy, values)


def test_aligned_dtype() -> None:
    """The values are given in the precision asked for, double by default."""
    a = na.ScalarArray(np.arange(3, dtype=np.float32), axes="x") * u.m

    assert _operators.aligned(a, shape=shape, unit=u.m).dtype == np.float64
    assert _operators.aligned(a, shape=shape, unit=u.m, dtype=np.float32).dtype == (
        np.float32
    )


def test_frames() -> None:
    """
    The axes of the frames are those of an array beyond its own, in the
    order the array has them, and none of them can be an axis of the other
    kind of array.
    """
    shape_own = dict(x=3, y=2)
    shape_other = dict(channel=4, z=5)

    assert _operators.frames(dict(y=2, x=3), shape_own, shape_other) == dict()
    assert _operators.frames(dict(x=3), shape_own, shape_other) == dict()

    result = _operators.frames(dict(time=6, x=3, y=2, repeat=7), shape_own, shape_other)

    assert result == dict(time=6, repeat=7)
    assert tuple(result) == ("time", "repeat")

    assert _operators.frames(dict(x=3, y=2, z=5), shape_own, shape_other) is None
    assert _operators.frames(dict(channel=4), shape_own, shape_other) is None


@pytest.mark.parametrize(
    argnames="combine",
    argvalues=[_operators._combine, _operators._combine.py_func],
    ids=["compiled", "python"],
)
def test__combine(combine: Callable) -> None:
    """
    The kernel sums the spectral lines into the signal of every pixel and
    the variance of its shot noise, each kept from being negative once it is
    summed, and adds the variance of the read noise.
    """
    rng = np.random.default_rng(0)
    photons = rng.uniform(-2, 2, size=(3, 2, 5, 4))
    gain = rng.uniform(1, 2, size=(3, 2))
    gain_variance = rng.uniform(1, 2, size=(3, 2))
    variance_read = rng.uniform(0, 1, size=(3, 5))

    images = np.empty((3, 5, 4))
    variance = np.empty((3, 5, 4))

    combine(photons, gain, gain_variance, variance_read, images, variance)

    signal = (photons * gain[..., np.newaxis, np.newaxis]).sum(axis=1)
    shot = (photons * gain_variance[..., np.newaxis, np.newaxis]).sum(axis=1)

    assert np.any(signal < 0)
    assert np.any(signal > 0)
    assert np.allclose(images, np.maximum(signal, 0), rtol=1e-14, atol=0)
    assert np.allclose(
        variance,
        np.maximum(shot, 0) + variance_read[..., np.newaxis],
        rtol=1e-14,
        atol=0,
    )


@pytest.mark.parametrize(
    argnames="snr_squared",
    argvalues=[_operators._snr_squared, _operators._snr_squared.py_func],
    ids=["compiled", "python"],
)
def test__snr_squared(snr_squared: Callable) -> None:
    """
    The kernel sums each row of the matrix over the voxels of one pixel of
    the scene, which gives what that pixel alone brings to a pixel of the
    sensor, and adds the square of its signal over the variance of its noise
    to the result, unless it brings nothing or the pixel has no noise.
    """
    rng = np.random.default_rng(0)
    num_channel, num_line, num_pixel = 2, 2, 5
    num_spectral, num_xy, num_frame = 6, 4, 2
    num_row = num_channel * num_line * num_pixel
    num_voxel = num_channel * num_spectral * num_xy

    # a matrix most of whose entries are empty, with one empty row
    matrix = rng.uniform(0, 1, size=(num_row, num_voxel))
    matrix[rng.uniform(size=matrix.shape) < 0.6] = 0
    matrix[3] = 0

    indptr = np.concatenate([[0], np.cumsum((matrix != 0).sum(axis=1))])
    rows, indices = np.nonzero(matrix)
    data = matrix[rows, indices]

    # some voxels send less than no light, and some pixels have no noise
    leaving = rng.uniform(-1, 2, size=(num_voxel, num_frame))
    gain = rng.uniform(1, 2, size=(num_channel, num_line))
    gain_variance = rng.uniform(1, 2, size=(num_channel, num_line))
    gain_variance[0, 0] = 0
    variance_read = rng.uniform(0, 1, size=(num_channel, num_pixel))
    variance_read[0, :2] = 0

    result = np.full((num_channel, num_line, num_xy, num_frame), 0.5)

    snr_squared(
        indptr,
        indices,
        data,
        leaving,
        gain,
        gain_variance,
        variance_read,
        result,
    )

    alone = np.einsum(
        "clpsx,sxf->clpxf",
        matrix.reshape(num_channel, num_line, num_pixel, -1, num_xy),
        leaving.reshape(-1, num_xy, num_frame),
    )
    signal = gain[..., np.newaxis, np.newaxis, np.newaxis] * alone
    variance = gain_variance[..., np.newaxis, np.newaxis, np.newaxis] * alone
    variance = variance + variance_read[:, np.newaxis, :, np.newaxis, np.newaxis]
    measured = (alone > 0) & (variance > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(measured, np.square(signal) / variance, 0)
    expected = 0.5 + expected.sum(axis=2)

    assert np.any(alone < 0)
    assert np.any((alone > 0) & (variance == 0))
    assert np.allclose(result, expected, rtol=1e-13, atol=0)
