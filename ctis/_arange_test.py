import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis


@pytest.mark.parametrize(
    argnames="start,stop,step",
    argvalues=[
        (0, 10, 3),  # range not divisible by step
        (0, 9, 3),  # range divisible by step
        (-10 * u.arcsec, 10 * u.arcsec, 6 * u.arcsec),  # astropy Quantity
        (500 * u.nm, 600 * u.nm, 7 * u.nm),  # astropy Quantity
    ],
)
def test_arange(
    start: float | u.Quantity,
    stop: float | u.Quantity,
    step: float | u.Quantity,
):
    result = ctis.arange(start, stop, axis="x", step=step)

    assert isinstance(result, na.AbstractScalarArray)
    assert result.axes == ("x",)
    assert na.unit_normalized(result) == na.unit_normalized(start)

    # the samples are spaced by `step`
    diff = np.diff(result, axis="x")
    assert np.allclose(diff, step)

    # the samples are centered within the range: the gaps at the two ends are
    # equal, and they are smaller than a full step
    gap_lower = result.min("x") - start
    gap_upper = stop - result.max("x")
    assert np.allclose(gap_lower, gap_upper)
    assert np.all(gap_lower >= 0 * step)
    assert np.all(gap_lower < step)

    # the samples fit within the range
    assert np.all(result >= start)
    assert np.all(result <= stop)


@pytest.mark.parametrize(
    argnames="wrap",
    argvalues=[
        lambda x: x,  # plain Quantity components
        lambda x: na.ScalarArray(x),  # ScalarArray components
    ],
    ids=["quantity", "scalararray"],
)
def test_arange_vector(wrap):
    start = na.Cartesian2dVectorArray(x=wrap(-10 * u.arcsec), y=wrap(-8 * u.arcsec))
    stop = na.Cartesian2dVectorArray(x=wrap(10 * u.arcsec), y=wrap(8 * u.arcsec))
    step = na.Cartesian2dVectorArray(x=wrap(6 * u.arcsec), y=wrap(5 * u.arcsec))
    axis = na.Cartesian2dVectorArray("x", "y")

    result = ctis.arange(start, stop, axis=axis, step=step)

    assert isinstance(result, na.Cartesian2dVectorArray)

    # the components form an outer-product grid, one axis per component
    assert set(na.shape(result)) == {"x", "y"}

    # each component is spaced by its own step and centered in its own range
    for c in ["x", "y"]:
        component = getattr(result, c)
        ax = getattr(axis, c)
        lower = getattr(start, c)
        upper = getattr(stop, c)
        pitch = getattr(step, c)

        assert np.allclose(np.diff(component, axis=ax), pitch)

        gap_lower = component.min(ax) - lower
        gap_upper = upper - component.max(ax)
        assert np.allclose(gap_lower, gap_upper)
        assert np.all(gap_lower >= 0 * pitch)
        assert np.all(gap_lower < pitch)
