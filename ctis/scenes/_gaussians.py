from typing import TypeVar
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "random_gaussians",
]

SpectralPositionalVectorT = TypeVar(
    name="SpectralPositionalVectorT",
    bound=na.AbstractSpectralPositionalVectorArray,
)


def _gaussian(
    inputs: na.AbstractSpectralPositionalVectorArray,
    amplitude: u.Quantity | na.AbstractScalar,
    center: na.AbstractSpectralPositionalVectorArray,
    width: na.AbstractSpectralPositionalVectorArray,
) -> na.AbstractScalar:
    """
    Compute a Gaussian kernel.

    Parameters
    ----------
    inputs
        The grid on which to evaluate the Gaussian.
    amplitude
        The height of the Gausian distribution.
    center
        The mean of the Gaussian distribution.
    width
        The standard deviation of the Gaussian distribution.
    """

    arg = -np.square(((inputs - center) / width).length) / 2
    return amplitude * np.exp(arg)


def random_gaussians(
    inputs: SpectralPositionalVectorT,
    width: na.AbstractSpectralPositionalVectorArray,
) -> na.FunctionArray[SpectralPositionalVectorT, na.ScalarArray]:
    r"""
    A scene with a randomly-positioned Gaussian distributions originally
    prepared by Amy Winebarger.

    Parameters
    ----------
    axis
        The names of the logical axes representing each
        :math:`\lambda,x,y` coordinate,
    num
        The number of pixels along each coordinate:

    Examples
    --------

    Plot this scene for the input grid and Gaussian widths originally
    prepared by Amy Winebarger.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import ctis

        # Define the spatial plate scale of the instrument
        platescale = 0.59 * u.arcsec / u.pix

        # Define the grid of positions and velocities on which to evaluate the
        # test pattern
        inputs = na.SpectralPositionalVectorArray(
            wavelength=na.linspace(-500, 500, axis="wavelength", num=101) * u.km / u.s,
            position=na.Cartesian2dVectorLinearSpace(
                start=-20 * platescale * u.pix,
                stop=20 * platescale * u.pix,
                axis=na.Cartesian2dVectorArray("x", "y"),
                num=41,
            ),
        )

        # Define the standard deviations of the Gaussians in space and velocity
        width = na.SpectralPositionalVectorArray(
            wavelength=27 * u.km / u.s,
            position=2.4 / 2.35 * u.arcsec,
        )

        # Compute the scene of random Gaussians for the
        # given input grid and standard deviations
        scene = ctis.scenes.random_gaussians(
            inputs=inputs,
            width=width,
        )

        # Plot the result
        with astropy.visualization.quantity_support():
            fig, axs = plt.subplots(
                ncols=2,
                gridspec_kw=dict(width_ratios=[.9,.1]),
                constrained_layout=True,
            )
            colorbar = na.plt.rgbmesh(
                C=scene,
                axis_wavelength="wavelength",
                ax=axs[0],
                vmin=0,
                vmax=scene.outputs.max(),
            )
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb="wavelength",
                ax=axs[1],
            )
            axs[1].yaxis.tick_right()
            axs[1].yaxis.set_label_position("right")
    """

    amplitude = np.array(
        [
            3.38,
            6,
            4,
            3.38,
            5,
            7,
            3.7,
            5.5,
        ]
    )

    center_x = np.array(
        [
            -2.5,
            2.5,
            0,
            0,
            0,
            -4,
            0,
            4,
        ]
    )

    center_y = np.array(
        [
            0,
            0,
            -2,
            2,
            4,
            4,
            -4,
            -4,
        ]
    )

    center_v = np.array(
        [
            +200,
            -200,
            -150,
            +150,
            -100,
            +100,
            -100,
            -150,
        ]
    )

    axis = "event"

    amplitude = na.ScalarArray(amplitude, axis)
    center_x = na.ScalarArray(center_x, axis) << u.arcsec
    center_y = na.ScalarArray(center_y, axis) << u.arcsec
    center_v = na.ScalarArray(center_v, axis) << (u.km / u.s)

    center = na.SpectralPositionalVectorArray(
        wavelength=center_v,
        position=na.Cartesian2dVectorArray(center_x, center_y),
    )

    outputs = _gaussian(
        inputs=inputs,
        amplitude=amplitude,
        center=center,
        width=width,
    )

    outputs = outputs.sum(axis)

    return na.FunctionArray(
        inputs=inputs,
        outputs=outputs,
    )
