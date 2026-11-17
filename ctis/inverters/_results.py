import abc
import dataclasses
import numpy as np
import matplotlib.axis
import matplotlib.pyplot as plt
import matplotlib.ticker
import astropy.units as u
import named_arrays as na
import ctis

__all__ = [
    "InversionResult",
]


def _percentile(a: na.AbstractScalar, q: float) -> u.Quantity:
    """
    A percentile of the elements of an array which are numbers, or zero if
    none of them is, such as the median of a scene which emits nothing.
    """
    if not np.any(np.isfinite(a)).ndarray:
        return 0 * na.unit_normalized(a)
    return np.nanpercentile(a, q).ndarray


def _scale(
    limits: tuple[u.Quantity, u.Quantity],
) -> tuple[float, float, u.UnitBase, int]:
    """
    The limits of a scale as numbers, the unit they are in, and the power of
    ten which is factored out of the scale, into its label, if they are far
    from one.
    """
    vmin, vmax = (u.Quantity(na.as_named_array(a).ndarray) for a in limits)

    # a plain zero is a zero of any unit, as it is for astropy
    plain = tuple(
        a.unit == u.dimensionless_unscaled and a.value == 0 for a in (vmin, vmax)
    )
    unit = vmax.unit if plain[0] else vmin.unit
    vmin = 0.0 if plain[0] else float(vmin.to_value(unit))
    vmax = 0.0 if plain[1] else float(vmax.to_value(unit))

    exponent = 0
    magnitude = max(abs(vmin), abs(vmax))
    if magnitude > 0 and not 1e-3 <= magnitude < 1e5:
        exponent = int(np.floor(np.log10(magnitude)))

    return vmin, vmax, unit, exponent


def _label(name: str, unit: u.UnitBase, exponent: int = 0) -> str:
    """
    The label of a scale: its name, and the power of ten and the unit which
    its numbers are in.
    """
    factors = []
    if exponent != 0:
        factors.append(f"$10^{{{exponent}}}$")
    if unit != u.dimensionless_unscaled:
        factors.append(f"{unit:latex_inline}")
    suffix = f" ({' '.join(factors)})" if factors else ""
    return f"{name}{suffix}"


def _values(a: na.AbstractScalar, unit: u.UnitBase) -> na.AbstractScalar:
    """An array as bare numbers in a unit."""
    return (a / unit).to(u.dimensionless_unscaled).value


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

    def _validate_truth(
        self,
        truth: na.FunctionArray[
            na.AbstractDopplerPositionalVectorArray,
            na.ScalarArray,
        ],
    ) -> None:
        """
        Check that the units of a true scene can be converted to those of
        the reconstructed scene, which comparing the two needs.

        Parameters
        ----------
        truth
            The true scene to check.
        """
        unit_truth = na.unit_normalized(truth.outputs)
        unit_solution = na.unit_normalized(self.solution.outputs)
        if not unit_truth.is_equivalent(unit_solution):
            raise ValueError(
                f"The true scene is in units of {unit_truth}, which cannot "
                f"be converted to those of the reconstructed scene, "
                f"{unit_solution}, to compare the two."
            )

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
            self._validate_truth(truth)
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

        # the scene which the default limits of the color scales come from
        radiance, median, iqr = next(iter(moments.values()))

        # the limits are percentiles, since otherwise a few pixels would set
        # the scale of all the others
        if range_radiance is None:
            bound = _percentile(radiance, 99.5)
            range_radiance = (0 * bound, bound)
        if range_median is None:
            bound = _percentile(unblanked(np.abs(median)), 99)
            range_median = (-bound, bound)
        if range_iqr is None:
            bound = _percentile(unblanked(iqr), 99)
            range_iqr = (0 * bound, bound)

        # every color scale is in the unit of its limits
        scales = [_scale(r) for r in (range_radiance, range_median, range_iqr)]
        labels = ("radiance", "median", "IQR")

        # the radiance and the interquartile range are magnitudes, and the
        # median has a sign
        color_blank = "0.55"
        cmaps = (
            plt.get_cmap("gray"),
            plt.get_cmap("RdBu_r").with_extremes(bad=color_blank),
            plt.get_cmap("Purples_r").with_extremes(bad=color_blank),
        )

        positions = dict()
        for name in scenes:
            position = scenes[name].inputs.position.explicit
            unit_x = na.unit_normalized(position.x)
            unit_y = na.unit_normalized(position.y)
            positions[name] = (
                _values(position.x, unit_x),
                _values(position.y, unit_y),
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
                    C=_values(moment, unit) / factor,
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
                        label=_label(labels[col], unit, exponent),
                    )
                    colorbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
                    colorbar.update_ticks()

                if row == num_rows - 1:
                    ax.set_xlabel(_label("scene $x$", unit_x))

            axs[row, 0].set_ylabel(f"{name}\n" + _label("scene $y$", unit_y))

        return fig, axs

    def plot_moments_histogram(
        self,
        truth: na.FunctionArray[
            na.AbstractDopplerPositionalVectorArray,
            na.ScalarArray,
        ],
        axis: None | str = None,
        num_bins: int = 50,
        range_radiance: None | tuple[None | u.Quantity, None | u.Quantity] = None,
        range_median: None | tuple[None | u.Quantity, None | u.Quantity] = None,
        range_iqr: None | tuple[None | u.Quantity, None | u.Quantity] = None,
        range_radiance_error: None | tuple[None | float, None | float] = None,
        range_median_error: None | tuple[None | u.Quantity, None | u.Quantity] = None,
        range_iqr_error: None | tuple[None | u.Quantity, None | u.Quantity] = None,
        percentile_radiance: float = 0,
        threshold_snr: None | float = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot column-normalized 2D histograms of the error of the
        reconstructed radiance, median, and interquartile range of the
        spectral line profile, pixel by pixel.

        The result is the figure and a 2D array of its axes, with a column
        for each of the three moments.
        The upper row is the error of each moment against its true value,
        and the lower row is the same error against the signal-to-noise
        ratio of the reconstructed line profile,
        the :meth:`~ctis.instruments.AbstractInstrument.snr` of the
        instrument for the reconstructed scene.

        The error of the radiance is the ratio of the reconstructed radiance
        to the true one, on a logarithmic scale, and the error of the median
        and of the interquartile range is the reconstructed moment minus the
        true one.
        A dashed line marks no error.
        In the upper row, a dotted line marks the error of a reconstruction
        which is the same in every pixel, at the median of the reconstructed
        moment, and so says nothing about the true scene:
        the closer the errors lie to that line,
        the less of the moment the reconstruction has recovered.

        Every column of a histogram is divided by the number of its pixels,
        those with that true moment or that signal-to-noise ratio,
        so it is the distribution of their errors.
        A pixel whose error is beyond the limits of the vertical scale is
        still counted, so a column can sum to less than one.
        By default, those limits are symmetric about no error, at three
        times the scatter of the errors of the pixels in the upper row,
        which is half the width of the central 68% of them.

        The signal-to-noise ratio is on a logarithmic scale, from one,
        below which the noise is larger than the signal, or from its 1st
        percentile if that is larger, to its 99th percentile.
        It is that of the reconstructed scene, the one which an inversion of
        observed images has, so it rises with the reconstructed radiance:
        of two pixels with the same true radiance, the one which is
        reconstructed brighter has the higher ratio.

        Every panel of the upper row is labeled with the Pearson correlation
        coefficient of the true and the reconstructed moment, the median of
        the error, and its scatter.
        For the radiance, the last two are those of the logarithm of the
        ratio, as percentages.

        Use :meth:`plot_moments` to plot maps of the same three moments.

        Parameters
        ----------
        truth
            The true scene which will be compared to the reconstructed scene.
            Its radiance is in units which can be converted to those of the
            reconstructed scene.
        axis
            The axis along which to compute the median and the interquartile
            range.
            The default is the wavelength axis of the instrument, which the
            radiance is integrated along whichever this is.
        num_bins
            The number of bins along each axis of every histogram.
        range_radiance
            The limits of the true radiance, the horizontal scale of the
            upper left panel, which is logarithmic, so the lower limit is
            positive.
            The scale is in the unit of its lower limit, or in that of its
            upper limit if the lower one is :obj:`None` or a plain zero,
            and so are those of the other two moments.
            A limit which is :obj:`None` is the 1st or the 99th percentile
            of the pixels in the upper row, since otherwise a few pixels
            would set the scale of all the others, and the lower limit is
            then no further than three decades below the upper one.
        range_median
            The limits of the true median, the horizontal scale of the
            upper middle panel.
            A limit which is :obj:`None` is the 1st or the 99th percentile
            of the pixels in the upper row.
        range_iqr
            The limits of the true interquartile range, the horizontal scale
            of the upper right panel.
            A limit which is :obj:`None` is the 1st or the 99th percentile
            of the pixels in the upper row.
        range_radiance_error
            The limits of the error of the radiance, the vertical scale of
            the left panels, which is logarithmic, so the lower limit is
            positive.
            A limit which is :obj:`None` is that of a scale which is
            symmetric about no error, at three times the scatter of the
            errors of the pixels in the upper row.
            Limits which are given keep the scale the same from one figure
            to the next, to compare two reconstructions.
        range_median_error
            The limits of the error of the median, the vertical scale of the
            middle panels, which is in the unit of the true median,
            whichever unit these limits are in.
            A limit which is :obj:`None` is that of a scale which is
            symmetric about no error, as for the radiance.
        range_iqr_error
            The limits of the error of the interquartile range, the vertical
            scale of the right panels, which is in the unit of the true
            interquartile range, whichever unit these limits are in.
            A limit which is :obj:`None` is that of a scale which is
            symmetric about no error, as for the radiance.
        percentile_radiance
            Spatial locations where the true radiance is at or below this
            percentile of it are left out of both rows.
            The default leaves out only the faintest pixels.
        threshold_snr
            Spatial locations where the signal-to-noise ratio of the
            reconstructed line profile is at or below this threshold are
            left out of the upper row, as :meth:`plot_moments` leaves them
            blank.
            The lower row keeps them, and marks the threshold with a dotted
            line.
            The default is to not leave out any pixels.

        Raises
        ------
        ValueError
            If the units of the true scene cannot be converted to those of
            the reconstructed scene,
            if the lower limit of the radiance or of its error is not
            positive,
            or if the instrument cannot be expressed as a pair of sparse
            matrices, which its signal-to-noise ratio is read from.
        """
        instrument = self.inverter.instrument
        solution = self.solution

        self._validate_truth(truth)

        moments_truth = self._moments(truth, axis)
        moments_solution = self._moments(solution, axis)

        snr = instrument.snr(solution)

        # the pixels of the lower row, and those of the upper row
        radiance_truth = moments_truth[0]
        where_lower = radiance_truth > np.nanpercentile(
            radiance_truth,
            percentile_radiance,
        )
        where_upper = where_lower
        if threshold_snr is not None:
            where_upper = where_lower & (snr > threshold_snr)

        def selected(
            a: na.AbstractScalar,
            where: na.AbstractScalar,
        ) -> na.AbstractScalar:
            """An array where a pixel is plotted, and no number elsewhere."""
            return np.where(where, a, np.nan)

        def statistics(error: na.AbstractScalar) -> tuple[float, float]:
            """
            The median of the errors which are numbers, and their scatter:
            half the width of the central 68% of them.
            """
            if not np.any(np.isfinite(error)).ndarray:
                return np.nan, np.nan
            lower, median, upper = (
                float(np.nanpercentile(error, q).ndarray) for q in (16, 50, 84)
            )
            return median, (upper - lower) / 2

        def histogram(
            x: na.AbstractScalar,
            y: na.AbstractScalar,
            bins_x: na.AbstractScalar,
            bins_y: na.AbstractScalar,
            where: na.AbstractScalar,
        ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
            """
            A histogram of `y` against `x`, each column of which is divided
            by the number of its pixels, whether `y` is within its bins or
            not.
            """
            where = where & np.isfinite(x)
            result = na.histogram2d(
                x,
                y,
                bins=na.Cartesian2dVectorArray(bins_x, bins_y),
                weights=where,
            )
            total = na.histogram(x, bins=bins_x, weights=where)

            # a column with no pixels is zero over zero
            with np.errstate(divide="ignore", invalid="ignore"):
                result.outputs = np.nan_to_num(result.outputs / total.outputs)

            return result

        def given(
            limits: None | tuple[None | u.Quantity, None | u.Quantity],
            unit: None | u.UnitBase = None,
        ) -> tuple[None | float, None | float, None | u.UnitBase]:
            """
            The limits of a scale as numbers, each of which is :obj:`None`
            if it is not given, and the unit they are in: the one they are
            converted to, or the one they are given in.
            """
            lower, upper = (None, None) if limits is None else limits
            if lower is None and upper is None:
                return None, None, unit

            # a limit which is not given is a plain zero for now, which
            # leaves the unit of the scale to the other limit
            vmin, vmax, unit_given, _ = _scale(
                (0 if lower is None else lower, 0 if upper is None else upper),
            )
            if vmin == 0 and vmax == 0:
                # a zero is all that is given, which has no unit of its own
                unit_given = unit

            factor = 1.0 if unit is None else unit_given.to(unit)

            return (
                None if lower is None else factor * vmin,
                None if upper is None else factor * vmax,
                unit_given if unit is None else unit,
            )

        def scale_log(axis: matplotlib.axis.Axis) -> None:
            """
            Label a logarithmic scale with plain numbers if it spans less
            than two decades, which powers of ten would label only once or
            twice.
            """
            lower, upper = axis.get_view_interval()
            span = upper / lower
            if span < 100:
                if span < 2.5:
                    locator = matplotlib.ticker.MaxNLocator(
                        nbins=5,
                        steps=[1, 2, 5, 10],
                    )
                elif span < 10:
                    locator = matplotlib.ticker.LogLocator(
                        subs=(1, 1.5, 2, 3, 5, 7),
                    )
                else:
                    locator = matplotlib.ticker.LogLocator(subs=(1, 2, 5))
                axis.set_major_locator(locator)
                axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                axis.set_minor_locator(matplotlib.ticker.NullLocator())

        # the limits of the horizontal scales are percentiles, since
        # otherwise a few pixels would set the scale of all the others
        percentiles = (1, 99)

        # the signal-to-noise ratio is plotted from one, below which the
        # noise is larger than the signal, unless every ratio is larger
        snr_lower = selected(snr, where_lower)
        min_snr = max(1.0, float(_percentile(snr_lower, percentiles[0])))
        max_snr = float(_percentile(snr_lower, percentiles[1]))
        if not min_snr < max_snr:
            # the signal of no pixel is larger than its noise
            max_snr = 10 * min_snr
        bins_snr = na.geomspace(min_snr, max_snr, axis="snr", num=num_bins + 1)

        names = ("radiance", "median", "IQR")
        ranges = (range_radiance, range_median, range_iqr)
        ranges_error = (range_radiance_error, range_median_error, range_iqr_error)

        kwargs_none = dict(color="tab:red", linestyle="dashed")
        kwargs_reference = dict(color="white", linestyle="dotted")
        kwargs_legend = dict(
            fontsize="small",
            facecolor="0.35",
            labelcolor="white",
            framealpha=0.9,
        )

        fig, axs = plt.subplots(
            nrows=2,
            ncols=len(names),
            constrained_layout=True,
            figsize=(10, 7.5),
        )

        for col, name in enumerate(names):
            # the radiance spans decades and its errors are relative, so it
            # is on a logarithmic scale and its error is a ratio
            log = col == 0

            vmin, vmax, unit = given(ranges[col])
            if unit is None:
                unit = na.unit_normalized(moments_truth[col])

            true = _values(moments_truth[col], unit)
            reconstructed = _values(moments_solution[col], unit)

            # a limit which is not given is that of the pixels
            default = vmin is None or vmax is None
            true_upper = selected(true, where_upper)
            if vmax is None:
                vmax = float(_percentile(true_upper, percentiles[1]))
            if vmin is None:
                vmin = float(_percentile(true_upper, percentiles[0]))
                if log:
                    vmin = max(vmin, vmax / 1000)
            elif log and not vmin > 0:
                raise ValueError(
                    f"The radiance is on a logarithmic scale, so the lower "
                    f"limit of `range_radiance` should be positive, but it "
                    f"is {range_radiance[0]}."
                )

            if default and not vmin < vmax:
                # every pixel in the upper row has the same true moment, or
                # there are none
                if log:
                    vmin, vmax = (vmin / 2, 2 * vmax) if vmin > 0 else (1.0, 10.0)
                else:
                    vmin, vmax = vmin - 0.5, vmax + 0.5

            if log:
                emin, emax, _ = given(ranges_error[col], u.dimensionless_unscaled)
                if emin is not None and not emin > 0:
                    raise ValueError(
                        f"The error of the radiance is on a logarithmic "
                        f"scale, so the lower limit of "
                        f"`range_radiance_error` should be positive, but it "
                        f"is {range_radiance_error[0]}."
                    )

                # a pixel which emits nothing has no ratio
                with np.errstate(divide="ignore", invalid="ignore"):
                    error = reconstructed / true
                error = np.where(np.isfinite(error) & (error > 0), error, np.nan)
                bias, scatter = statistics(selected(np.log(error), where_upper))
                half = 3 * scatter if scatter > 0 else np.log(2)
                limits_error = (float(np.exp(-half)), float(np.exp(half)))
                label_error = f"reconstructed / true {name}"
                # adding zero turns a negative zero into a positive one
                text = (
                    f"median error ${np.round(100 * np.expm1(bias)) + 0.0:+.0f}$%\n"
                    f"scatter ${100 * scatter:.0f}$%"
                )
            else:
                emin, emax, _ = given(ranges_error[col], unit)

                error = reconstructed - true
                bias, scatter = statistics(selected(error, where_upper))
                half = 3 * scatter if scatter > 0 else 1.0
                limits_error = (-half, half)
                label_error = _label(f"reconstructed $-$ true {name}", unit)
                suffix = ""
                if unit != u.dimensionless_unscaled:
                    suffix = f" {unit:latex_inline}"
                text = (
                    f"median error ${bias:+.3g}${suffix}\n"
                    f"scatter ${scatter:.3g}${suffix}"
                )

            # the limits of the errors which are given
            limits_error = (
                limits_error[0] if emin is None else emin,
                limits_error[1] if emax is None else emax,
            )

            space = na.geomspace if log else na.linspace
            bins_true = space(vmin, vmax, axis="true", num=num_bins + 1)
            bins_error = space(*limits_error, axis="error", num=num_bins + 1)

            rows = (
                (true, bins_true, where_upper),
                (snr, bins_snr, where_lower),
            )
            for row, (x, bins_x, where) in enumerate(rows):
                ax = axs[row, col]

                hist = histogram(x, error, bins_x, bins_error, where)
                img = na.plt.pcolormesh(
                    C=hist,
                    ax=ax,
                    vmax=np.percentile(hist.outputs, 99.5),
                )
                colorbar = fig.colorbar(
                    img.ndarray.item(),
                    ax=ax,
                    location="top",
                    label="probability",
                )
                colorbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
                colorbar.update_ticks()

                ax.axhline(1 if log else 0, label="no error", **kwargs_none)

                # the signal-to-noise ratio is on a logarithmic scale too
                if log or row == 1:
                    ax.set_xscale("log")
                if log:
                    ax.set_yscale("log")
                ax.set_xlim(bins_x.min().ndarray, bins_x.max().ndarray)
                ax.set_ylim(*limits_error)
                if log or row == 1:
                    scale_log(ax.xaxis)
                if log:
                    scale_log(ax.yaxis)
                ax.set_ylabel(label_error)

            ax_true, ax_snr = axs[:, col]

            # where a reconstruction which is the same in every pixel lies
            constant = float(_percentile(selected(reconstructed, where_upper), 50))
            if not log:
                ax_true.axline(
                    (constant, 0),
                    slope=-1,
                    label="constant reconstruction",
                    **kwargs_reference,
                )
            elif constant > 0:
                ax_true.axline(
                    (constant, 1),
                    (10 * constant, 0.1),
                    label="constant reconstruction",
                    **kwargs_reference,
                )

            # the correlation of fewer than two pixels is not defined
            finite = where_upper & np.isfinite(true) & np.isfinite(reconstructed)
            r = np.nan
            if finite.sum().ndarray > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = na.stats.pearsonr(true, reconstructed, where=finite)
                r = float(r.ndarray)

            ax_true.text(
                x=0.96,
                y=0.96,
                s=f"Pearson's $r = {r:.3f}$\n{text}",
                transform=ax_true.transAxes,
                ha="right",
                va="top",
                fontsize="small",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
            ax_true.set_xlabel(_label(f"true {name}", unit))

            if threshold_snr is not None:
                ax_snr.axvline(threshold_snr, label="threshold", **kwargs_reference)
            ax_snr.set_xlabel("signal-to-noise ratio")

        axs[0, 0].legend(loc="lower left", **kwargs_legend)
        if threshold_snr is not None:
            axs[1, 0].legend(loc="lower right", **kwargs_legend)

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
