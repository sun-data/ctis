import numpy as np
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
    A version of :func:`named_arrays.regridding.weights` tailored for use with
    the spatial-spectral cubes in this package.

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

    _weights_2d, shape_input, shape_output = na.regridding.weights(
        coordinates_input=position_input,
        coordinates_output=position_output,
        axis_input=axis_position_input,
        axis_output=axis_position_output,
        method="conservative",
    )


def _weights_3d(
    weights_2d: na.ScalarArray,
    shape_inputs: dict[str, int],
    shape_outputs: dict[str, int],
    axis_wavelength: str,
) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:

    pass


def weights_3d_numpy(
    weights_2d: np.ndarray,
    shape_inputs: tuple[int, ...],
    shape_outputs: tuple[int, ...],
    axis_wavelength: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:

    pass
