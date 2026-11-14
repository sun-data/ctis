from typing import Any, Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
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
    argnames="a",
    argvalues=[
        na.ScalarArray(np.ones(4), axes="z") * u.m,
        na.ScalarArray(np.ones(4), axes="x") * u.m,
        na.ScalarArray(np.ones(3), axes="x") * u.s,
        na.ScalarArray(np.ones(3), axes="x"),
        na.UniformUncertainScalarArray(1 * u.m, width=1 * u.m, num_distribution=3),
        na.Cartesian2dVectorArray(1, 2) * u.m,
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
def test_aligned_unsupported(a: Any) -> None:
    """An array which cannot be expressed as a plain array is left alone."""
    assert _operators.aligned(a, shape=shape, unit=u.m) is None


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

    assert _operators.vertices(coordinates, shape=shape, unit=u.m) is None


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


def test_regridder_unit() -> None:
    """
    Weights with a dimensionless unit are assembled into a matrix without
    one, scaled by the unit, so that its products are plain arrays.
    """
    shape_input = dict(channel=3, x=4, y=5)
    shape_output = dict(channel=3, u=6, v=4)

    weights_input = na.ScalarArray(np.full((4, 5), 50.0), axes=("x", "y")) * u.percent

    matrix = _operators.regridder(
        weights=_weights(weights_input),
        shape_input=shape_input,
        shape_output=shape_output,
    )
    unweighted = _operators.regridder(
        weights=_weights(),
        shape_input=shape_input,
        shape_output=shape_output,
    )

    rng = np.random.default_rng(0)
    values = rng.uniform(size=tuple(shape_input.values()))

    result = matrix(values)

    assert matrix.unit is None
    assert not isinstance(result, u.Quantity)
    assert result.sum() > 0
    assert np.allclose(result, unweighted(values) / 2, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    argnames="shape_input,shape_output",
    argvalues=[
        (dict(channel=3, x=4, y=5), dict(channel=3, u=6, v=4)),
        (dict(channel=1, x=4, y=5), dict(channel=3, u=6, v=4)),
        (dict(channel=3, x=4, y=5), dict(channel=1, u=6, v=4)),
    ],
    ids=["each", "shared", "summed"],
)
def test_regridder(
    shape_input: dict[str, int],
    shape_output: dict[str, int],
) -> None:
    """
    The matrix gives what the weights give, with an input shared by the
    channels if it has only one, and summed over the channels if the output
    has only one.
    """
    weights = _weights()

    rng = np.random.default_rng(0)
    values = rng.uniform(size=tuple(shape_input.values()))

    result = _operators.regridder(
        weights=weights,
        shape_input=shape_input,
        shape_output=shape_output,
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


@pytest.mark.parametrize(
    argnames="shape_input,shape_output",
    argvalues=[
        (dict(x=4, y=5, channel=3), dict(channel=3, u=6, v=4)),
        (dict(channel=3, y=5, x=4), dict(channel=3, u=6, v=4)),
        (dict(channel=3, x=4, y=5), dict(u=6, v=4, channel=3)),
        (dict(channel=3, x=4, y=5), dict(channel=3, v=4, u=6)),
    ],
)
def test_regridder_axes(
    shape_input: dict[str, int],
    shape_output: dict[str, int],
) -> None:
    """The orthogonal axes come first, and the resampled axes in grid order."""
    with pytest.raises(ValueError, match="axes"):
        _operators.regridder(
            weights=_weights(),
            shape_input=shape_input,
            shape_output=shape_output,
        )


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
