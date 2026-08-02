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

    Each input voxel is weighted by its volume (the wavelength bin width times
    the solid angle subtended by each field pixel) before the conservative
    regridding and divided by the output voxel volume afterward, so the
    *integral* of the field is preserved and ``values_input`` is treated as a
    density (such as a spectral radiance) rather than a per-voxel total. Both
    steps use :func:`named_arrays.regridding.regrid` with
    ``method="conservative"``.

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

        # Define the vertices of the (fine) input grid.
        coordinates_input = na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=9) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=+10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("x", "y"),
                num=13,
            ),
        )

        # Define the vertices of the (coarser) output grid.
        coordinates_output = na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=5) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=+10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("x", "y"),
                num=7,
            ),
        )

        # Define a random cube sampled on the input voxel centers.
        values_input = na.random.uniform(
            low=0,
            high=1,
            shape_random=dict(wavelength=8, x=12, y=12),
        )

        # Resample the cube onto the coarser output grid.
        values_output = ctis.regrid(
            coordinates_input=coordinates_input,
            coordinates_output=coordinates_output,
            values_input=values_input,
            axis_wavelength="wavelength",
            axis_position=("x", "y"),
        )

        # Plot the input and output grids side by side, with wavelength
        # represented by color. The resampling preserves the field density, so
        # a shared normalization lets one colorbar describe both panels.
        vmax = values_output.max()
        wavelength_min = coordinates_input.wavelength.min()
        wavelength_max = coordinates_input.wavelength.max()
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=3,
                gridspec_kw=dict(width_ratios=[0.45, 0.45, 0.1]),
                sharex=False,
                figsize=(9, 4),
                constrained_layout=True,
            )
            colorbar = na.plt.rgbmesh(
                coordinates_input,
                C=values_input,
                axis_wavelength="wavelength",
                vmin=0,
                vmax=vmax,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                ax=ax[0],
            )
            na.plt.rgbmesh(
                coordinates_output,
                C=values_output,
                axis_wavelength="wavelength",
                vmin=0,
                vmax=vmax,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                ax=ax[1],
            )
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb="wavelength",
                ax=ax[2],
            )
            ax[2].yaxis.tick_right()
            ax[2].yaxis.set_label_position("right")
            ax[0].set_title("input grid")
            ax[1].set_title("output grid")
    """

    axis = (axis_wavelength, *axis_position)

    # normalize to plain spectral-positional vectors so grids defined on a
    # Doppler vector (as produced by the `IdealInstrument`) are accepted.
    coordinates_input = coordinates_input.spectral_positional
    coordinates_output = coordinates_output.spectral_positional

    # weight each input voxel by its volume so the conservative regridding
    # preserves the integral of the field rather than the per-voxel sum.
    values = values_input * coordinates_input.volume_cell(axis)

    # 1D conservative interpolation along the wavelength axis.
    values = na.regridding.regrid(
        coordinates_input=coordinates_input.wavelength,
        coordinates_output=coordinates_output.wavelength,
        values_input=values,
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

    # recover the field density on the output grid by dividing out the output
    # voxel volume.
    values = values / coordinates_output.volume_cell(axis)

    return values
