import abc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import astropy.units as u
import astropy.visualization
import named_arrays as na
import ctis

__all__ = [
    "InversionResult",
]


@dataclasses.dataclass
class AbstractInversionResult(
    abc.ABC,
):
    """An interface describing the results of an inversion attempt."""

    @property
    @abc.abstractmethod
    def solution(
        self,
    ) -> na.FunctionArray[
        na.AbstractDopplerPositionalVectorArray,
        na.ScalarArray,
    ]:
        """The reconstructed scene found by the inversion."""

    @property
    @abc.abstractmethod
    def success(self) -> bool:
        """Whether the inversion was successful."""

    @property
    @abc.abstractmethod
    def images(
        self,
    ) -> na.FunctionArray[
        na.AbstractDopplerPositionalVectorArray,
        na.ScalarArray,
    ]:
        """The observed images used to calculate the inversion."""

    @property
    @abc.abstractmethod
    def inverter(self) -> "ctis.inverters.AbstractInverter":
        """The inversion algorithm instance that produced these results."""

    @property
    @abc.abstractmethod
    def message(self) -> str:
        """Any message from the inverter regarding these results."""

    def _moments(
        self,
        scene: na.FunctionArray[
            na.AbstractDopplerPositionalVectorArray,
            na.ScalarArray,
        ],
        axis: None | str = None,
    ) -> tuple[na.AbstractScalar, na.AbstractScalar, na.AbstractScalar]:
        """
        The radiance, the median, and the interquartile range of the spectral
        line profile in every pixel of a scene.

        Parameters
        ----------
        scene
            The scene to compute the moments of, the reconstructed scene or
            one on the same grid.
        axis
            The axis along which to compute the median and the interquartile
            range, which is the wavelength axis of the instrument by default.
            The radiance is integrated along the wavelength axis of the
            instrument whichever this is.
        """
        axis_wavelength = self.inverter.instrument.axis_wavelength
        if axis is None:
            axis = axis_wavelength

        dw = scene.inputs.wavelength.volume_cell(axis_wavelength)

        radiance = (scene.outputs * dw).sum(axis_wavelength)
        median = na.pdf.median(
            x=scene.inputs.velocity,
            f=scene.outputs,
            axis=axis,
        )
        iqr = na.pdf.iqr(
            x=scene.inputs.velocity,
            f=scene.outputs,
            axis=axis,
        )

        return radiance, median, iqr

    def plot_moments(
        self,
        truth: (
            None
            | na.FunctionArray[
                na.AbstractDopplerPositionalVectorArray,
                na.ScalarArray,
            ]
        ) = None,
        axis: None | str = None,
        range_radiance: None | tuple[u.Quantity, u.Quantity] = None,
        range_median: None | tuple[u.Quantity, u.Quantity] = None,
        range_iqr: None | tuple[u.Quantity, u.Quantity] = None,
        threshold_snr: None | float = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot maps of the radiance, median, and interquartile range of the
        spectral line profile in every pixel of the reconstructed scene.

        The result is the figure and a 2D array of its axes, with a row for
        each scene and a column for each of the three moments.

        Use :meth:`plot_moments_histogram` to compare the moments of the
        reconstructed scene to those of a true scene pixel by pixel.

        Parameters
        ----------
        truth
            An optional true scene, which is plotted in a row above the
            reconstructed scene, on the same color scales.
            Its radiance is in units which can be converted to those of the
            reconstructed scene.
        axis
            The axis along which to compute the median and the interquartile
            range.
            The default is the wavelength axis of the instrument, which the
            radiance is integrated along whichever this is.
        range_radiance
            The limits of the color scale of the radiance.
            The scale is in the unit of its lower limit, or in that of its
            upper limit if the lower one is a plain zero, and so are those
            of the other two moments.
            The default is zero to the 99.5th percentile.
        range_median
            The limits of the color scale of the median.
            The default is symmetric about zero, and reaches the 99th
            percentile of the magnitude of the median.
        range_iqr
            The limits of the color scale of the interquartile range.
            The default is zero to the 99th percentile.
        threshold_snr
            The median and the interquartile range are left blank where the
            signal-to-noise ratio of the reconstructed line profile is at or
            below this threshold, since they mean little there.
            The ratio is that of each pixel of the reconstructed scene as if
            it were alone in the scene,
            the :meth:`~ctis.instruments.AbstractInstrument.snr` of the
            instrument for the reconstructed scene, which is only available
            for an instrument which can be expressed as a pair of sparse
            matrices.
            The default is to leave no pixels blank.

        Raises
        ------
        ValueError
            If a scene has an axis beyond those of one scene, such as the
            reconstruction of several frames, which these maps cannot show,
            or if the units of the true scene cannot be converted to those
            of the reconstructed scene.
        """
        instrument = self.inverter.instrument
        solution = self.solution

        scenes = dict(reconstructed=solution)
        if truth is not None:
            unit_truth = na.unit_normalized(truth.outputs)
            unit_solution = na.unit_normalized(solution.outputs)
            if not unit_truth.is_equivalent(unit_solution):
                raise ValueError(
                    f"The true scene is in units of {unit_truth}, which "
                    f"cannot be converted to those of the reconstructed "
                    f"scene, {unit_solution}, to plot the two on the same "
                    f"color scales."
                )
            scenes = dict(true=truth) | scenes

        moments = {name: self._moments(scenes[name], axis) for name in scenes}

        for name in scenes:
            shape = na.shape_broadcasted(*moments[name])
            axes = tuple(ax for ax in shape if ax not in instrument.axis_scene_xy)
            if axes:
                raise ValueError(
                    f"These are maps of one scene, with the axes "
                    f"{instrument.axis_scene_xy}, but the moments of the "
                    f"{name} scene also have the axes {axes}."
                )

        where = None
        if threshold_snr is not None:
            where = instrument.snr(solution) > threshold_snr

        def blanked(a: na.AbstractScalar) -> na.AbstractScalar:
            if where is None:
                return a
            return np.where(where, a, np.nan)

        def unblanked(a: na.AbstractScalar) -> na.AbstractScalar:
            """
            A moment where it is not blank, or everywhere if every pixel is
            blank.
            """
            b = blanked(a)
            if np.any(np.isfinite(b)).ndarray:
                return b
            return a

        def percentile(a: na.AbstractScalar, q: float) -> u.Quantity:
            """
            A percentile of the pixels which are numbers, or zero if none of
            them is, such as the median of a scene which emits nothing.
            """
            if not np.any(np.isfinite(a)).ndarray:
                return 0 * na.unit_normalized(a)
            return np.nanpercentile(a, q).ndarray

        # the scene which the default limits of the color scales come from
        radiance, median, iqr = next(iter(moments.values()))

        # the limits are percentiles, since otherwise a few pixels would set
        # the scale of all the others
        if range_radiance is None:
            bound = percentile(radiance, 99.5)
            range_radiance = (0 * bound, bound)
        if range_median is None:
            bound = percentile(unblanked(np.abs(median)), 99)
            range_median = (-bound, bound)
        if range_iqr is None:
            bound = percentile(unblanked(iqr), 99)
            range_iqr = (0 * bound, bound)

        def scale(
            limits: tuple[u.Quantity, u.Quantity],
        ) -> tuple[float, float, u.UnitBase, int]:
            """
            The limits of a color scale as numbers, the unit they are in, and
            the power of ten which is factored out of the scale, into its
            label, if they are far from one.
            """
            vmin, vmax = (u.Quantity(na.as_named_array(a).ndarray) for a in limits)

            # a plain zero is a zero of any unit, as it is for astropy
            plain = tuple(
                a.unit == u.dimensionless_unscaled and a.value == 0
                for a in (vmin, vmax)
            )
            unit = vmax.unit if plain[0] else vmin.unit
            vmin = 0.0 if plain[0] else float(vmin.to_value(unit))
            vmax = 0.0 if plain[1] else float(vmax.to_value(unit))

            exponent = 0
            magnitude = max(abs(vmin), abs(vmax))
            if magnitude > 0 and not 1e-3 <= magnitude < 1e5:
                exponent = int(np.floor(np.log10(magnitude)))

            return vmin, vmax, unit, exponent

        # every color scale is in the unit of its limits
        scales = [scale(r) for r in (range_radiance, range_median, range_iqr)]
        labels = ("radiance", "median", "IQR")

        # the radiance and the interquartile range are magnitudes, and the
        # median has a sign
        color_blank = "0.55"
        cmaps = (
            plt.get_cmap("gray"),
            plt.get_cmap("RdBu_r").with_extremes(bad=color_blank),
            plt.get_cmap("Purples_r").with_extremes(bad=color_blank),
        )

        def label(name: str, unit: u.UnitBase, exponent: int = 0) -> str:
            factors = []
            if exponent != 0:
                factors.append(f"$10^{{{exponent}}}$")
            if unit != u.dimensionless_unscaled:
                factors.append(f"{unit:latex_inline}")
            suffix = f" ({' '.join(factors)})" if factors else ""
            return f"{name}{suffix}"

        def values(a: na.AbstractScalar, unit: u.UnitBase) -> na.AbstractScalar:
            return (a / unit).to(u.dimensionless_unscaled).value

        positions = dict()
        for name in scenes:
            position = scenes[name].inputs.position.explicit
            unit_x = na.unit_normalized(position.x)
            unit_y = na.unit_normalized(position.y)
            positions[name] = (
                values(position.x, unit_x),
                values(position.y, unit_y),
                unit_x,
                unit_y,
            )

        # the panels have the proportions of the scene
        x, y, _, _ = next(iter(positions.values()))
        aspect = (y.max() - y.min()) / (x.max() - x.min())
        aspect = float(aspect.ndarray)

        num_rows = len(scenes)

        fig, axs = plt.subplots(
            nrows=num_rows,
            ncols=len(labels),
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True,
            figsize=(10, 1.4 + num_rows * (0.3 + 2.7 * aspect)),
        )

        for row, name in enumerate(scenes):
            x, y, unit_x, unit_y = positions[name]

            for col, moment in enumerate(moments[name]):
                ax = axs[row, col]

                vmin, vmax, unit, exponent = scales[col]
                factor = 10.0**exponent

                if col > 0:
                    moment = blanked(moment)

                img = na.plt.pcolormesh(
                    x,
                    y,
                    C=values(moment, unit) / factor,
                    ax=ax,
                    cmap=cmaps[col],
                    vmin=vmin / factor,
                    vmax=vmax / factor,
                )

                ax.set_aspect("equal")

                if row == 0:
                    colorbar = fig.colorbar(
                        img.ndarray.item(),
                        ax=axs[:, col],
                        location="top",
                        label=label(labels[col], unit, exponent),
                    )
                    colorbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
                    colorbar.update_ticks()

                if row == num_rows - 1:
                    ax.set_xlabel(label("scene $x$", unit_x))

            axs[row, 0].set_ylabel(f"{name}\n" + label("scene $y$", unit_y))

        return fig, axs

    def plot_moments_histogram(
        self,
        truth: na.FunctionArray[
            na.AbstractDopplerPositionalVectorArray,
            na.ScalarArray,
        ],
        axis: None | str = None,
        num_bins: int = 50,
        range_radiance: None | tuple[u.Quantity, u.Quantity] = None,
        range_median: None | tuple[u.Quantity, u.Quantity] = None,
        range_iqr: None | tuple[u.Quantity, u.Quantity] = None,
        percentile_radiance: float = 0,
        threshold_snr: None | float = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot column-normalized 2D histograms of the true vs. reconstructed
        radiance, median, and interquartile range of the spectral line profile.

        Use :meth:`plot_moments` to plot maps of the same three moments.

        Parameters
        ----------
        truth
            The true scene which will be compared to the reconstructed scene.
        axis
            The axis along which to compute the median and the interquartile
            range.
            The default is the wavelength axis of the instrument, which the
            radiance is integrated along whichever this is.
        num_bins
            The number of bins along each axis of the histogram.
        range_radiance
            The domain of the radiance histogram.
        range_median
            The domain of the median histogram.
        range_iqr
            The domain of the interquartile range histogram.
        percentile_radiance
            Spatial locations below this threshold are excluded from the histogram.
            The default is to not exclude any pixels.
        threshold_snr
            Spatial locations where the signal-to-noise ratio of the
            reconstructed line profile,
            the :meth:`~ctis.instruments.AbstractInstrument.snr` of the
            instrument for the reconstructed scene, is at or below this
            threshold are excluded from the histogram,
            as :meth:`plot_moments` leaves them blank.
            The default is to not exclude any pixels.
        """

        recon = self.solution

        radiance_truth, median_truth, iqr_truth = self._moments(truth, axis)
        radiance_recon, median_recon, iqr_recon = self._moments(recon, axis)

        thresh = np.nanpercentile(radiance_truth, percentile_radiance)
        where = radiance_truth > thresh

        if threshold_snr is not None:
            where = where & (self.inverter.instrument.snr(recon) > threshold_snr)

        r_radiance = na.stats.pearsonr(
            x=radiance_truth,
            y=radiance_recon,
            where=where & np.isfinite(radiance_recon),
        ).ndarray
        r_median = na.stats.pearsonr(
            x=median_truth,
            y=median_recon,
            where=where & np.isfinite(median_recon),
        ).ndarray
        r_iqr = na.stats.pearsonr(
            x=iqr_truth,
            y=iqr_recon,
            where=where & np.isfinite(iqr_recon),
        ).ndarray

        bins = dict(true=num_bins, reconstructed=num_bins)

        if range_radiance is None:
            range_radiance = (None, None)

        if range_median is None:
            range_median = (None, None)

        if range_iqr is None:
            range_iqr = (None, None)

        min_radiance, max_radiance = range_radiance
        min_median, max_median = range_median
        min_iqr, max_iqr = range_iqr

        if min_radiance is None:
            min_radiance = 0 * radiance_truth.unit
        if max_radiance is None:
            max_radiance = radiance_truth.max()

        if min_median is None:
            min_median = np.nanmin(median_truth)
        if max_median is None:
            max_median = np.nanmax(median_truth)

        if min_iqr is None:
            min_iqr = 0 * iqr_truth.unit
        if max_iqr is None:
            max_iqr = iqr_truth.max()

        hist_radiance = na.histogram2d(
            radiance_truth,
            radiance_recon,
            bins=bins,
            min=min_radiance,
            max=max_radiance,
            weights=where,
        )
        hist_median = na.histogram2d(
            median_truth,
            median_recon,
            bins=bins,
            min=min_median,
            max=max_median,
            weights=where,
        )
        hist_iqr = na.histogram2d(
            iqr_truth,
            iqr_recon,
            bins=bins,
            min=min_iqr,
            max=max_iqr,
            weights=where,
        )

        # columns which contain no pixels are normalized to NaN,
        # and then replaced with zero below.
        with np.errstate(divide="ignore", invalid="ignore"):
            hist_radiance = hist_radiance / hist_radiance.sum("reconstructed")
            hist_median = hist_median / hist_median.sum("reconstructed")
            hist_iqr = hist_iqr / hist_iqr.sum("reconstructed")

        hist_radiance.outputs = np.nan_to_num(
            x=hist_radiance.outputs,
            posinf=0,
            neginf=0,
        )
        hist_median.outputs = np.nan_to_num(hist_median.outputs)
        hist_iqr.outputs = np.nan_to_num(hist_iqr.outputs)

        with astropy.visualization.quantity_support():
            fig, axs = plt.subplots(
                constrained_layout=True,
                figsize=(10, 4),
                ncols=3,
            )
            ax_radiance, ax_median, ax_iqr = axs
            img_radiance = na.plt.pcolormesh(
                C=hist_radiance,
                ax=ax_radiance,
                vmax=np.nanpercentile(hist_radiance.outputs, 99.5),
            )
            img_median = na.plt.pcolormesh(
                C=hist_median,
                ax=ax_median,
                vmax=np.nanpercentile(hist_median.outputs, 99.5),
            )
            img_iqr = na.plt.pcolormesh(
                C=hist_iqr,
                ax=ax_iqr,
                vmax=np.nanpercentile(hist_iqr.outputs, 99.5),
            )

            pt_radiance = np.nanmean(radiance_truth).ndarray.value
            pt_median = np.nanmean(median_truth).ndarray.value
            pt_iqr = np.nanmean(iqr_truth).ndarray.value

            ax_radiance.axline(
                (pt_radiance, pt_radiance),
                slope=1,
                color="tab:red",
                linestyle="dashed",
            )
            ax_median.axline(
                (pt_median, pt_median),
                slope=1,
                color="tab:red",
                linestyle="dashed",
            )
            ax_iqr.axline(
                (pt_iqr, pt_iqr),
                slope=1,
                color="tab:red",
                linestyle="dashed",
            )
            plt.colorbar(
                img_radiance.ndarray.item(),
                ax=ax_radiance,
                location="top",
                label="probability",
            )
            plt.colorbar(
                img_median.ndarray.item(),
                ax=ax_median,
                location="top",
                label="probability",
            )
            plt.colorbar(
                img_iqr.ndarray.item(),
                ax=ax_iqr,
                location="top",
                label="probability",
            )
            ax_radiance.set_xlabel(
                f"true radiance ({ax_radiance.get_xlabel()})",
            )
            ax_radiance.set_ylabel(
                f"reconstructed radiance ({ax_radiance.get_ylabel()})",
            )
            ax_median.set_xlabel(
                f"true median ({ax_median.get_xlabel()})",
            )
            ax_median.set_ylabel(
                f"reconstructed median ({ax_median.get_ylabel()})",
            )
            ax_iqr.set_xlabel(
                f"true IQR ({ax_iqr.get_xlabel()})",
            )
            ax_iqr.set_ylabel(
                f"reconstructed IQR ({ax_iqr.get_ylabel()})",
            )

            ax_radiance.text(
                x=0.05,
                y=0.95,
                s=f"Pearson's $r = {r_radiance:.03f}$",
                transform=ax_radiance.transAxes,
                ha="left",
                va="top",
                color="white",
            )
            ax_median.text(
                x=0.05,
                y=0.95,
                s=f"Pearson's $r = {r_median:.03f}$",
                transform=ax_median.transAxes,
                ha="left",
                va="top",
                color="white",
            )
            ax_iqr.text(
                x=0.05,
                y=0.95,
                s=f"Pearson's $r = {r_iqr:.03f}$",
                transform=ax_iqr.transAxes,
                ha="left",
                va="top",
                color="white",
            )

            ax_radiance.set_aspect("equal")
            ax_median.set_aspect("equal")
            ax_iqr.set_aspect("equal")

        return fig, axs


@dataclasses.dataclass
class InversionResult(
    AbstractInversionResult,
):
    """
    The results of an inversion attempt.
    """

    solution: na.FunctionArray[
        na.AbstractDopplerPositionalVectorArray, na.ScalarArray
    ] = dataclasses.MISSING
    """The reconstructed scene found by the inversion."""

    success: bool = dataclasses.MISSING
    """A boolean flag indicating whether the inversion was successful."""

    images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray] = (
        dataclasses.MISSING
    )
    """The observed images on which the inversion was performed."""

    inverter: "ctis.inverters.AbstractInverter" = dataclasses.MISSING
    """The inversion algorithm instance that produced these results."""

    message: str = dataclasses.MISSING
    """Any message from the inversion routine concerning the results."""
