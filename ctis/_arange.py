"""
Evenly-spaced sequences for building CTIS coordinate grids.
"""

import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "arange",
]


def arange(
    start: "float | u.Quantity | na.AbstractVectorArray",
    stop: "float | u.Quantity | na.AbstractVectorArray",
    axis: "str | na.AbstractVectorArray",
    step: "float | u.Quantity | na.AbstractVectorArray" = 1,
) -> "na.AbstractArray":
    """
    Return evenly-spaced values over a range, centered within it.

    This is the centered counterpart of :func:`named_arrays.arange`, which
    starts exactly at `start` and leaves the remainder of the range as a gap on
    the right. :func:`ctis.arange` fits as many samples spaced by `step` as
    possible into ``[start, stop]`` and centers them, splitting the leftover
    evenly between the two ends. This is convenient for building a coordinate
    grid with a fixed pitch that is centered on a field of view rather than
    biased toward one edge.

    Unlike :func:`named_arrays.arange`, this is built on
    :func:`named_arrays.linspace`, so it works correctly when the arguments are
    :class:`astropy.units.Quantity` instances.

    If `start`, `stop`, and `step` are instances of
    :class:`named_arrays.AbstractVectorArray`, each component is centered
    independently along its own axis (given by the matching component of
    `axis`), producing an outer-product grid.

    Parameters
    ----------
    start
        The lower bound of the range.
    stop
        The upper bound of the range. The samples are centered within
        ``[start, stop]``, so `start` and `stop` are samples only when `step`
        divides the range evenly.
    axis
        The name of the new logical axis of the result. If `start`, `stop`, and
        `step` are vectors, this should be a vector of axis names, one per
        component.
    step
        The spacing between adjacent samples.

    Examples
    --------

    :func:`named_arrays.arange` starts at `start`, leaving the remainder of the
    range as a gap on the right.

    .. jupyter-execute::

        import named_arrays as na
        import ctis

        na.arange(0, 10, axis="x", step=3)

    :func:`ctis.arange` uses the same samples and step, but splits that
    remainder evenly between the two ends of the range.

    .. jupyter-execute::

        ctis.arange(0, 10, axis="x", step=3)

    Vector arguments produce a centered grid, with each component sampled
    independently along its own axis.

    .. jupyter-execute::

        import astropy.units as u

        ctis.arange(
            start=na.Cartesian2dVectorArray(-10, -8) * u.arcsec,
            stop=na.Cartesian2dVectorArray(10, 8) * u.arcsec,
            axis=na.Cartesian2dVectorArray("x", "y"),
            step=na.Cartesian2dVectorArray(6, 5) * u.arcsec,
        )
    """
    # the number of samples spaced by `step` that fit within the range,
    # computed per-component so vector arguments produce a grid.
    num = na.as_named_array((stop - start) / step)
    num = (np.floor(num + 1e-10) + 1).astype(int)
    if not isinstance(num, na.AbstractVectorArray):
        # na.linspace requires a Python int for the count of a scalar axis
        num = int(num.ndarray)

    # center the samples by splitting the leftover evenly between both ends
    span = (num - 1) * step
    offset = (stop - start - span) / 2

    return na.linspace(start + offset, stop - offset, axis=axis, num=num)
