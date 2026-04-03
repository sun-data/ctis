import numpy as np
import numba
import named_arrays as na

__all__ = [
    "weights"
]


def weights(
    wavelength_input: na.AbstractScalar,
    position_input: na.AbstractCartesian2dVectorArray,
    position_output: na.AbstractCartesian2dVectorArray,
    axis_wavelength_input: str,
    axis_position_input: tuple[str, str],
    axis_position_output: tuple[str, str],
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    A version of :func:`named_arrays.regridding.weights` tailored for use with
    the spatial-spectral cubes in this package.

    Conservatively resampling spatial-spectral cubes is a 3D problem,
    which can be reduced to a 2D problem :math:`+` a 1D problem since the
    wavelength coordinates do not vary along the spatial coordinates.

    Parameters
    ----------
    wavelength_input
    position_input
    position_output
    axis_wavelength_input
    axis_position_input
    axis_position_output
    """

    wavelength_input = wavelength_input.cell_centers(axis_wavelength_input)
    position_input = position_input.cell_centers(axis_wavelength_input)

    _weights_2d, shape_input, shape_output = na.regridding.weights(
        coordinates_input=position_input,
        coordinates_output=position_output,
        axis_input=axis_position_input,
        axis_output=axis_position_output,
        method="conservative",
    )

    return




def _weights_3d(
    weights: na.ScalarArray,
    shape_inputs: dict[str, int],
    shape_outputs: dict[str, int],
    axis_wavelength_input: str,
    axis_position_input: tuple[str, str],
    axis_position_output: tuple[str, str],
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:

    axes = weights.axes

    result = _weights_3d_numpy(
        weights=weights.ndarray,
        axis_wavelength=axes.index(axis_wavelength_input),
    )

    result = na.ScalarArray(
        ndarray=result,
        axes=tuple(ax for ax in axes if ax != axis_wavelength_input),
    )

    shape_inputs[axis_wavelength_input] = shape_outputs.pop(axis_wavelength_input)

    return result, shape_inputs, shape_outputs


def _weights_3d_numpy(
    weights: np.ndarray,
    # shape_inputs: tuple[int, ...],
    # shape_outputs: tuple[int, ...],
    axis_wavelength: int,
    # axis_position_input: tuple[int, int],
    # axis_position_output: tuple[str, str],
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:

    num_wavelength = weights.shape[axis_wavelength]

    weights = np.moveaxis(
        a=weights,
        source=axis_wavelength,
        destination=~0,
    )

    shape_result = weights.shape[:~0]

    weights = numba.typed.List(weights.reshape(-1))

    weights = _weights_3d_numba(
        weights=weights,
        num_wavelength=num_wavelength,
    )

    weights = np.array(weights).reshape(shape_result)

    return weights


@numba.njit()
def _weights_3d_numba(
    weights: numba.typed.List,
    num_wavelength: int,
) -> numba.typed.List:
    """
    Convert a set of 2d weights to 3d.

    Since :mod:`numba` doesn't accept arrays of objects,
    the 2D weights must be a list of lists of tuples i,j,w.

    It would be nice if the input could be a 3D list of tuples,
    but constructing this outside of :mod:`numba`-compiled functions is
    prohibitive.
    Instead, the user must flatten the first two axes and provide their
    shapes to reconstruct the 3D inputs.

    Parameters
    ----------
    weights


    Returns
    -------

    """

    result = numba.typed.List()
    result_t = numba.typed.List()

    for d in numba.prange(len(weights)):

        d = numba.types.int64(d)

        w = d % num_wavelength

        weights_tw = weights[d]

        for n in range(len(weights_tw)):

            i_input, i_output, weight = weights_tw[n]

            i_input = (num_wavelength - 1) * i_input + w

            result_t.append((i_input, i_output, weight))

        if w == (num_wavelength - 1):
            result.append(result_t)
            result_t = numba.typed.List()

    return result
