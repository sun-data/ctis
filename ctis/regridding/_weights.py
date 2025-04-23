from typing import Sequence
import named_arrays as na

__all__ = [

    "weights"
]


def weights(
    wavelength: na.AbstractScalar,
    position_input: na.AbstractCartesian2dVectorArray,
    position_output: na.AbstractCartesian2dVectorArray,
    axis_wavelength: str,
    axis_position_input: tuple[str, str],
    axis_position_output: tuple[str, str],
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
    """
    A version of :func:`named_arrays.weights` targeted for use with
    the spatial-spectral cubes used in this package.

    Parameters
    ----------
    wavelength
    position_input
    position_output
    axis_wavelength
    axis_position_input
    axis_position_output
    """

    wavelength = wavelength.cell_centers(axis_wavelength)
    position_input = position_input.cell_centers(axis_wavelength)
    position_output = position_output.cell_centers(axis_wavelength)

    _weights, shape_input, shape_output = na.regridding.weights(
        coordinates_input=position_input,
        coordinates_output=position_output,
        axis_input=axis_position_input,
        axis_output=axis_position_output,
        method="conservative",
    )


