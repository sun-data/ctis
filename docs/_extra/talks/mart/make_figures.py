"""
Generate the figures for the talk *Inverting a Synthetic Scene with MART*.

This script repeats the calculation in the ``simple-mart`` tutorial and saves
slide-sized figures into a ``figures/`` directory next to this file.
It is run by ``docs/conf.py`` on every documentation build,
so the slides always match the current version of :mod:`ctis`.
"""

import json
import pathlib
import numpy as np
import matplotlib.lines
import matplotlib.ticker
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
import ctis

directory = pathlib.Path(__file__).parent / "figures"
"""The directory where the figures are saved."""

seed = 42
"""The seed of the photon shot noise, so that the figures are reproducible."""

paper = "#f6f4ef"
"""The background color of the slides."""

ink = "#1f2430"
"""The color of the text on the slides."""

muted = "#5b6270"
"""The color of secondary text, like tick labels."""

blue = "#2f6db5"
"""The color of the original scene in line plots."""

orange = "#c8501e"
"""The color of the reconstructed scene in line plots."""

colors_channel = [blue, orange, "#2a9d8f", "#8a5cb8"]
"""The color of each channel in line plots."""

style = {
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "figure.facecolor": paper,
    "axes.facecolor": paper,
    "savefig.facecolor": paper,
    "font.size": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.frameon": False,
    "text.color": ink,
    "axes.labelcolor": ink,
    "axes.edgecolor": muted,
    "xtick.color": muted,
    "ytick.color": muted,
    "lines.linewidth": 3,
    "animation.embed_limit": 100,
}
"""
Matplotlib settings for slides.

A figure 1 inch wide occupies 100 pixels of a 1920-pixel-wide slide,
so a 20-point font is about 28 pixels tall on the slide.
"""


def main():

    directory.mkdir(exist_ok=True)

    velocity = na.linspace(-500, 500, axis="wavelength", num=21) * u.km / u.s
    wavelength_rest = 171 * u.AA

    position_scene = na.Cartesian2dVectorLinearSpace(
        start=-10 * u.arcsec,
        stop=10 * u.arcsec,
        axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
        num=na.Cartesian2dVectorArray(64 + 1, 64 + 1),
    )
    position_sensor = na.Cartesian2dVectorArray(
        x=na.arange(0, 128 + 1, axis="sensor_x") * u.pix,
        y=na.arange(0, 64 + 1, axis="sensor_y") * u.pix,
    )

    coordinates_scene = na.DopplerPositionalVectorArray.from_velocity(
        velocity=velocity,
        wavelength_rest=wavelength_rest,
        position=position_scene,
    )
    coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
        velocity=velocity,
        wavelength_rest=wavelength_rest,
        position=position_sensor,
    )

    scene = ctis.scenes.gaussians(coordinates_scene)
    scene = scene + scene.max() / 100

    angle = na.linspace(0, 360, num=4, axis="channel", endpoint=False) * u.deg
    angle = angle + 5.64 * u.deg

    dispersion_velocity = 10 * u.km / u.s / u.pix
    dispersion = dispersion_velocity * u.pix
    dispersion = dispersion.to(u.AA, equivalencies=u.doppler_optical(wavelength_rest))
    dispersion = (dispersion - wavelength_rest) / u.pix

    plate_scale = 0.4 * u.arcsec / u.pix

    instrument = ctis.instruments.IdealInstrument(
        area_effective=1 * u.cm**2,
        timedelta_exposure=20 * u.s,
        plate_scale=plate_scale,
        dispersion=dispersion,
        angle=angle,
        wavelength_ref=wavelength_rest,
        position_ref=na.Cartesian2dVectorArray(64, 32) * u.pix,
        coordinates_scene=coordinates_scene,
        coordinates_sensor=coordinates_sensor,
        channel=angle.to_string_array("%03d"),
        axis_channel="channel",
        axis_wavelength="wavelength",
        axis_scene_xy=("scene_x", "scene_y"),
        axis_sensor_xy=("sensor_x", "sensor_y"),
    )

    # `IdealInstrument.image()` draws unseeded shot noise for each wavelength
    # before summing over wavelength.
    # A sum of independent Poisson variables is itself Poisson,
    # so drawing the noise after the sum has the same distribution
    # and lets us fix the seed.
    images = instrument.image(scene, noise=False)
    quantum_yield = instrument.quantum_yield
    images = na.FunctionArray(
        inputs=images.inputs,
        outputs=na.random.poisson(images.outputs / quantum_yield, seed=seed)
        * quantum_yield,
    )

    mart = ctis.inverters.MartInverter(
        instrument=instrument,
        intermediate=True,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        inversion = mart(images)

    def wavelength_to_velocity(wavelength):
        wavelength = wavelength * u.AA
        equivalency = u.doppler_optical(wavelength_rest)
        return wavelength.to_value(u.km / u.s, equivalencies=equivalency)

    def velocity_to_wavelength(velocity):
        velocity = velocity * u.km / u.s
        equivalency = u.doppler_optical(wavelength_rest)
        return velocity.to_value(u.AA, equivalencies=equivalency)

    def label_channel(j):
        """The label of the `j`-th channel, its dispersion angle."""
        return f"{int(angle[dict(channel=j)].ndarray.value):03d}°"

    def style_scene(ax):
        """Label the axes of an image of the scene."""
        ax.set_aspect("equal")
        ax.set_xlabel("scene $x$ (arcsec)")
        ax.set_ylabel("scene $y$ (arcsec)")
        # leave the corners unlabeled so the x and y tick labels don't collide
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-5, 0, 5]))
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([-5, 0, 5]))

    def rgb(ax, cax, C, **kwargs):
        """Plot a false-color image of `C`, with velocity encoded as color."""
        colorbar = na.plt.rgbmesh(
            C=C,
            axis_wavelength="wavelength",
            ax=ax,
            vmin=0,
            vmax=scene.outputs.max(),
            **kwargs,
        )
        style_scene(ax)
        if cax is not None:
            colorbar_rgb(cax, colorbar)
        return colorbar

    def colorbar_rgb(cax, colorbar):
        """Plot the colorbar of :func:`rgb`, labeled in velocity."""
        na.plt.pcolormesh(
            C=colorbar,
            axis_rgb="wavelength",
            ax=cax,
        )
        cax.set_xticks([])
        cax.set_xlabel("brighter →", fontsize=16)
        cax.set_yticks([])
        cax.set_ylabel("")
        # match the height of the neighboring image, which has a square box
        width_ratios = cax.get_gridspec().get_width_ratios()
        cax.set_box_aspect(width_ratios[0] / width_ratios[-1])
        secondary = cax.secondary_yaxis(
            location="right",
            functions=(wavelength_to_velocity, velocity_to_wavelength),
        )
        secondary.set_ylabel("Doppler velocity (km/s)")

    def stairs_velocity(ax, spectrum, **kwargs):
        """Plot a spectrum against velocity, with wavelength on the top axis."""
        na.plt.stairs(
            velocity.to(u.km / u.s),
            spectrum.to(u.erg / (u.s * u.sr * u.cm**2 * u.AA)),
            ax=ax,
            baseline=None,
            linewidth=3,
            **kwargs,
        )
        ax.set_xlabel("Doppler velocity (km/s)")
        ax.set_ylabel(
            "average radiance\n(erg s$^{-1}$ sr$^{-1}$ cm$^{-2}$ Å$^{-1}$)",
        )
        ax.set_xlim(velocity.min().ndarray.value, velocity.max().ndarray.value)
        ax.set_ylim(bottom=0)
        secondary = ax.secondary_xaxis(
            location="top",
            functions=(velocity_to_wavelength, wavelength_to_velocity),
        )
        secondary.set_xlabel("wavelength (Å)", labelpad=12)
        ax.spines[["right"]].set_visible(False)

    spectrum = scene.outputs.mean(("scene_x", "scene_y"))
    spectrum_inverted = inversion.solution.outputs.mean(("scene_x", "scene_y"))

    with plt.rc_context(style), astropy.visualization.quantity_support():

        # The title slide: the scene with no axes
        fig, ax = plt.subplots(figsize=(8, 8))
        rgb(ax, None, scene)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(directory / "hero.png", transparent=True)
        plt.close(fig)

        # The test scene
        fig, axs = plt.subplots(
            ncols=2,
            figsize=(9.6, 7.6),
            gridspec_kw=dict(width_ratios=[0.88, 0.12]),
            layout="compressed",
        )
        fig.get_layout_engine().set(wspace=0.08)
        rgb(*axs, scene)
        fig.savefig(directory / "scene.png")
        plt.close(fig)

        # The lines of sight of each channel through the scene
        fig, axs = plt.subplots(
            ncols=2,
            figsize=(17, 7.6),
            sharey=True,
            constrained_layout=True,
        )
        slope = plate_scale / dispersion_velocity
        line = na.linspace(-40, 40, axis="line", num=17) * u.arcsec
        velocity_line = na.linspace(-500, 500, axis="v", num=2) * u.km / u.s
        for i, ax in enumerate(axs):
            component = "xy"[i]
            C = scene.outputs.sum(f"scene_{'yx'[i]}")
            position = getattr(scene.inputs.position, component)
            na.plt.pcolormesh(
                position[{f"scene_{'yx'[i]}": 0}],
                velocity,
                C=C.value,
                ax=ax,
                cmap="gray",
            )
            handles = []
            for j in (i, i + 2):
                angle_j = angle[dict(channel=j)]
                direction = np.cos(angle_j) if i == 0 else np.sin(angle_j)
                position_line = line - slope * direction * velocity_line
                na.plt.plot(
                    position_line.to(u.arcsec),
                    velocity_line,
                    ax=ax,
                    axis="v",
                    color=colors_channel[j],
                    linewidth=2,
                )
                handle = matplotlib.lines.Line2D(
                    [],
                    [],
                    color=colors_channel[j],
                    label=label_channel(j),
                )
                handles.append(handle)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-500, 500)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
            ax.set_xlabel(f"scene ${component}$ (arcsec)")
            ax.legend(
                handles=handles,
                title="channel",
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )
        axs[0].set_ylabel("Doppler velocity (km/s)")
        fig.savefig(directory / "lines_of_sight.png")
        plt.close(fig)

        # The average spectrum of the scene
        fig, ax = plt.subplots(figsize=(15, 7.4), constrained_layout=True)
        stairs_velocity(ax, spectrum, color=blue)
        fig.savefig(directory / "spectrum.png")
        plt.close(fig)

        # The images measured by each channel
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(17, 7.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        vmax = images.outputs.max().ndarray.value
        for j, ax in enumerate(axs.flat):
            img = na.plt.pcolormesh(
                images.inputs.position.x,
                images.inputs.position.y,
                C=images.outputs[dict(channel=j)].value,
                ax=ax,
                cmap="gray",
                vmin=0,
                vmax=vmax,
            )
            ax.set_aspect("equal")
            ax.set_title(f"channel {label_channel(j)}")
            ax.set_xlabel("")
            ax.set_ylabel("")
        for ax in axs[1]:
            ax.set_xlabel("sensor $x$ (pixels)")
        for ax in axs[:, 0]:
            ax.set_ylabel("sensor $y$ (pixels)")
        fig.colorbar(
            img.ndarray.item(),
            ax=axs,
            label=f"signal ({images.outputs.unit:latex_inline})",
        )
        fig.savefig(directory / "images.png")
        plt.close(fig)

        # The reconstruction at each iteration of MART
        chi2 = inversion.mean_chi_squared.mean(instrument.axis_channel)
        label = "iteration " + inversion.iteration.to_string_array("%d")
        label = label + r",   $\langle \chi^2 \rangle$ = "
        label = label + chi2.to_string_array("%.2f", format_unit="")

        def figure_reconstruction():
            fig, axs = plt.subplots(
                ncols=3,
                figsize=(17, 7.8),
                gridspec_kw=dict(width_ratios=[0.44, 0.44, 0.08]),
                layout="compressed",
            )
            rgb(axs[0], axs[2], scene)
            axs[0].set_title("original")
            axs[1].set_title("reconstructed")
            return fig, axs

        fig, axs = figure_reconstruction()
        rgb(axs[1], None, inversion.solution)
        axs[1].set_ylabel("")
        axs[1].text(
            x=0.5,
            y=-0.12,
            s=label[{mart.axis_iteration: -1}].ndarray,
            transform=axs[1].transAxes,
            ha="center",
            va="top",
        )
        fig.savefig(directory / "reconstruction.png")
        plt.close(fig)

        fig, axs = figure_reconstruction()
        ani, _ = na.plt.rgbmovie(
            label,
            scene.inputs.wavelength,
            scene.inputs.position.x,
            scene.inputs.position.y,
            C=inversion.solutions.outputs,
            axis_time=mart.axis_iteration,
            axis_wavelength="wavelength",
            ax=axs[1],
            vmin=0,
            vmax=scene.outputs.max(),
        )
        style_scene(axs[1])
        axs[1].set_ylabel("")
        ani.save(directory / "reconstruction.mp4", fps=4, dpi=120)
        plt.close(fig)

        # The convergence of MART
        fig, axs = plt.subplots(
            nrows=2,
            figsize=(15, 7.8),
            sharex=True,
            constrained_layout=True,
        )
        for j in range(instrument.num_channel):
            index = dict(channel=j)
            na.plt.plot(
                inversion.iteration,
                inversion.mean_chi_squared[index],
                ax=axs[0],
                axis=mart.axis_iteration,
                color=colors_channel[j],
                label=label_channel(j),
            )
            na.plt.plot(
                inversion.iteration,
                inversion.correlation_residual[index],
                ax=axs[1],
                axis=mart.axis_iteration,
                color=colors_channel[j],
            )
        axs[0].axhline(1, color=muted, linestyle="dashed", linewidth=2)
        axs[0].set_yscale("log")
        axs[0].set_ylabel(r"$\langle \chi^2 \rangle$")
        axs[0].legend(title="channel", loc="upper left", bbox_to_anchor=(1, 1))
        axs[1].set_ylabel("signal-correlated\nresidual")
        axs[1].set_xlabel("iteration")
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(directory / "convergence.png")
        plt.close(fig)

        # The average spectrum of the reconstruction
        fig, ax = plt.subplots(figsize=(15, 7.4), constrained_layout=True)
        stairs_velocity(ax, spectrum, color=blue, label="original")
        stairs_velocity(
            ax,
            spectrum_inverted,
            color=orange,
            label="reconstructed",
        )
        ax.legend(loc="upper right")
        fig.savefig(directory / "spectrum_comparison.png")
        plt.close(fig)

        # The line moments of every pixel
        with np.errstate(divide="ignore", invalid="ignore"):
            fig, axs = inversion.plot_moments(scene, axis="wavelength")
        fig.set_size_inches(17.6, 6.6)
        # separate the colorbar labels from their tick labels
        for cax in fig.axes[len(axs) :]:
            cax.xaxis.labelpad = 12
        fig.savefig(directory / "moments.png")
        plt.close(fig)

    def peak(spectrum, sign):
        """The peak of the blue- or red-shifted component of `spectrum`."""
        where = sign * velocity.cell_centers("wavelength") > 0
        return np.where(where, spectrum, 0 * spectrum).max().ndarray

    def gap(spectrum):
        """The average of `spectrum` between the two components."""
        where = np.abs(velocity.cell_centers("wavelength")) < 50 * u.km / u.s
        return ((spectrum * where).sum() / where.sum()).ndarray

    numbers = dict(
        num_unknown=int(scene.outputs.size),
        num_measurement=int(images.outputs.size),
        num_channel=int(instrument.num_channel),
        num_iteration=int(inversion.num_iteration),
        iteration_final=int(inversion.num_iteration - 1),
        chi2_final=float(chi2[{mart.axis_iteration: -1}].ndarray),
        dispersion=float(dispersion.to_value(u.mAA / u.pix)),
        peak_blue=float(peak(spectrum_inverted, -1) / peak(spectrum, -1)),
        peak_red=float(peak(spectrum_inverted, +1) / peak(spectrum, +1)),
        gap=float(gap(spectrum_inverted) / gap(spectrum)),
    )

    path = directory / "numbers.js"
    path.write_text(f"window.numbers = {json.dumps(numbers, indent=4)};\n")

    print(json.dumps(numbers, indent=4))


if __name__ == "__main__":
    main()
