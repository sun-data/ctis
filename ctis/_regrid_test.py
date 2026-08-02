import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis


def _grid(num_wavelength: int, num_position: int) -> na.SpectralPositionalVectorArray:
    return na.SpectralPositionalVectorArray(
        wavelength=na.linspace(500, 600, axis="wavelength", num=num_wavelength) * u.nm,
        position=na.Cartesian2dVectorLinearSpace(
            start=-10 * u.arcsec,
            stop=+10 * u.arcsec,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=num_position,
        ),
    )


def _grid_doppler(
    num_wavelength: int,
    num_position: int,
) -> na.DopplerPositionalVectorArray:
    return na.DopplerPositionalVectorArray(
        wavelength=na.linspace(500, 600, axis="wavelength", num=num_wavelength) * u.nm,
        wavelength_rest=550 * u.nm,
        position=na.Cartesian2dVectorLinearSpace(
            start=-10 * u.arcsec,
            stop=+10 * u.arcsec,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=num_position,
        ),
    )


@pytest.mark.parametrize(
    argnames="coordinates_input,coordinates_output",
    argvalues=[
        (_grid(5, 7), _grid(9, 13)),  # refine
        (_grid(9, 13), _grid(5, 7)),  # coarsen
    ],
)
def test_regrid(
    coordinates_input: na.SpectralPositionalVectorArray,
    coordinates_output: na.SpectralPositionalVectorArray,
):
    axis_wavelength = "wavelength"
    axis_position = ("x", "y")

    num_input = coordinates_input.shape
    values_input = na.random.uniform(
        low=0,
        high=1,
        shape_random={
            axis_wavelength: num_input[axis_wavelength] - 1,
            axis_position[0]: num_input[axis_position[0]] - 1,
            axis_position[1]: num_input[axis_position[1]] - 1,
        },
    )

    result = ctis.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        axis_wavelength=axis_wavelength,
        axis_position=axis_position,
    )

    assert isinstance(result, na.AbstractScalarArray)
    assert np.all(np.isfinite(result))

    # the result is sampled on the output voxel centers
    num_output = coordinates_output.shape
    assert na.shape(result) == {
        axis_wavelength: num_output[axis_wavelength] - 1,
        axis_position[0]: num_output[axis_position[0]] - 1,
        axis_position[1]: num_output[axis_position[1]] - 1,
    }

    # the input and output grids span the same volume, so the volume-weighted
    # conservative resampling preserves the integral of the field (to within
    # the tolerance of the perturbation applied by the 2D conservative step).
    axis = (axis_wavelength, *axis_position)
    integral_input = (values_input * coordinates_input.volume_cell(axis)).sum()
    integral_output = (result * coordinates_output.volume_cell(axis)).sum()
    ratio = float((integral_output / integral_input).ndarray)
    assert np.isclose(ratio, 1, rtol=0.05)


def test_regrid_doppler():
    # grids defined on a Doppler vector (as produced by the `IdealInstrument`)
    # are accepted, normalized to spectral-positional for the volume weighting.
    axis_wavelength = "wavelength"
    axis_position = ("x", "y")
    coordinates_input = _grid_doppler(9, 13)
    coordinates_output = _grid_doppler(5, 7)

    num_input = coordinates_input.shape
    values_input = na.random.uniform(
        low=0,
        high=1,
        shape_random={
            axis_wavelength: num_input[axis_wavelength] - 1,
            axis_position[0]: num_input[axis_position[0]] - 1,
            axis_position[1]: num_input[axis_position[1]] - 1,
        },
    )

    result = ctis.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        axis_wavelength=axis_wavelength,
        axis_position=axis_position,
    )

    assert isinstance(result, na.AbstractScalarArray)
    assert np.all(np.isfinite(result))

    num_output = coordinates_output.shape
    assert na.shape(result) == {
        axis_wavelength: num_output[axis_wavelength] - 1,
        axis_position[0]: num_output[axis_position[0]] - 1,
        axis_position[1]: num_output[axis_position[1]] - 1,
    }
