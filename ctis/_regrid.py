"""
Resample spectral cubes between spectral/spatial grids.
"""

import named_arrays as na

__all__ = [
    "regrid",
]


def regrid(
    coordinates_input: "na.AbstractSpectralPositionalVectorArray",
    coordinates_output: "na.AbstractSpectralPositionalVectorArray",
    values_input: "na.AbstractScalar",
    axis_wavelength: str,
    axis_position: tuple[str, str],
) -> "na.AbstractScalarArray":
    """
    Conservatively resample a spectral cube from one spectral/spatial grid onto
    another.

    The resampling is split into two independent conservative steps: a 1D
    conservative interpolation along the wavelength axis, followed by a 2D
    conservative interpolation along the two spatial axes. Separating the
    spectral and spatial resampling is much cheaper than a single 3D
    conservative regrid and is exact whenever the wavelength grid does not
    depend on spatial position (and vice versa), which is the usual case for a
    CTIS scene.

    Both steps use :func:`named_arrays.regridding.regrid` with
    ``method="conservative"``, so the total (the sum of ``values_input`` over
    the resampled axes) is preserved. ``values_input`` is therefore treated as
    an extensive quantity: a per-voxel total rather than a density. Multiply a
    spectral radiance by its voxel volume before calling this function if the
    integral of the radiance is what should be conserved.

    Parameters
    ----------
    coordinates_input
        The wavelength and position of the vertices of each voxel of the input
        grid.
    coordinates_output
        The wavelength and position of the vertices of each voxel of the output
        grid.
    values_input
        The value in each voxel of the input grid, sampled on the voxel centers.
    axis_wavelength
        The logical axis corresponding to changing wavelength coordinate.
        Shared by the input and output grids.
    axis_position
        The two logical axes corresponding to changing position coordinate.
        Shared by the input and output grids.

    Examples
    --------

    Resample a random spectral cube onto a finer grid.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import ctis

        # Define the vertices of the input grid.
        coordinates_input = na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=5) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=+10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("x", "y"),
                num=7,
            ),
        )

        # Define the vertices of the (finer) output grid.
        coordinates_output = na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=9) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=+10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("x", "y"),
                num=13,
            ),
        )

        # Define a random cube sampled on the input voxel centers.
        values_input = na.random.uniform(
            low=0,
            high=1,
            shape_random=dict(wavelength=4, x=6, y=6),
        )

        # Resample the cube onto the output grid.
        values_output = ctis.regrid(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            values_input=values_input,
            axis_wavelength="wavelength",
            axis_position=("x", "y"),
        )

        # Plot the input and output grids, with wavelength represented by color.
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=2,
                sharex=True,
                sharey=True,
                figsize=(8, 4),
                constrained_layout=True,
            )
            na.plt.rgbmesh(
                coordinates_input,
                C=values_input,
                axis_wavelength="wavelength",
                ax=ax[0],
            )
            na.plt.rgbmesh(
                coordinates_output,
                C=values_output,
                axis_wavelength="wavelength",
                ax=ax[1],
            )
            ax[0].set_title("input grid")
            ax[1].set_title("output grid")
    """

    # 1D conservative interpolation along the wavelength axis.
    values = na.regridding.regrid(
        coordinates_input=coordinates_input.wavelength,
        coordinates_output=coordinates_output.wavelength,
        values_input=values_input,
        axis_input=(axis_wavelength,),
        axis_output=(axis_wavelength,),
        method="conservative",
    )

    # 2D conservative interpolation along the spatial axes.
    values = na.regridding.regrid(
        coordinates_input=coordinates_input.position,
        coordinates_output=coordinates_output.position,
        values_input=values,
        axis_input=axis_position,
        axis_output=axis_position,
        method="conservative",
    )

    return values
