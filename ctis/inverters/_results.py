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


def _limits(
    limits: None | tuple[None | u.Quantity, None | u.Quantity],
    unit: None | u.UnitBase = None,
) -> tuple[None | float, None | float, None | u.UnitBase]:
    """
    The limits of a scale as numbers, each of which is :obj:`None` if it is
    not given, and the unit they are in.

    The unit is the one they are converted to, if there is one, and
    otherwise that of the first limit which has one, or :obj:`None` if
    neither has.
    A plain zero has none, since it is a zero of any unit, as it is for
    astropy.

    Parameters
    ----------
    limits
        The lower and the upper limit of the scale, or :obj:`None` if
        neither is given.
    unit
        An optional unit to convert the limits to.
    """
    if limits is None:
        limits = (None, None)

    quantities = [
        None if a is None else u.Quantity(na.as_named_array(a).ndarray) for a in limits
    ]
    plain = [
        q is not None and q.unit == u.dimensionless_unscaled and q.value == 0
        for q in quantities
    ]

    if unit is None:
        units = [q.unit for q, p in zip(quantities, plain) if q is not None and not p]
        if units:
            unit = units[0]

    vmin, vmax = (
        None if q is None else 0.0 if p else float(q.to_value(unit))
        for q, p in zip(quantities, plain)
    )

    return vmin, vmax, unit


def _blanked(
    a: na.AbstractScalar,
    where: None | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    An array where a condition holds, and no number elsewhere, or the array
    itself if there is no condition.
    """
    if where is None:
        return a
    return np.where(where, a, np.nan)


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
        range_radiance: None | tuple[None | u.Quantity, None | u.Quantity] = None,
        range_median: None | tuple[None | u.Quantity, None | u.Quantity] = None,
        range_iqr: None | tuple[None | u.Quantity, None | u.Quantity] = None,
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
            The scale is in the unit of the first limit which has one, as a
            plain zero has none, and so are those of the other two moments.
            The default is zero to the 99.5th percentile, and a limit which
            is :obj:`None` is that of the default, here and for the other
            two moments.
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

        def unblanked(a: na.AbstractScalar) -> na.AbstractScalar:
            """
            A moment where it is not blank, or everywhere if every pixel is
            blank.
            """
            b = _blanked(a, where)
            if np.any(np.isfinite(b)).ndarray:
                return b
            return a

        # the scene which the default limits of the color scales come from
        radiance, median, iqr = next(iter(moments.values()))

        # the default limits are percentiles, since otherwise a few pixels
        # would set the scale of all the others
        bound_radiance = _percentile(radiance, 99.5)
        bound_median = _percentile(unblanked(np.abs(median)), 99)
        bound_iqr = _percentile(unblanked(iqr), 99)
        defaults = (
            (0 * bound_radiance, bound_radiance),
            (-bound_median, bound_median),
            (0 * bound_iqr, bound_iqr),
        )
        ranges = (range_radiance, range_median, range_iqr)

        # every color scale is in the unit of its limits, or in that of the
        # moment if neither limit has one
        scales = []
        for limits, default, moment in zip(ranges, defaults, (radiance, median, iqr)):
            vmin, vmax, unit = _limits(limits)
            if unit is None:
                unit = na.unit_normalized(moment)
            if vmin is None:
                vmin = float(u.Quantity(default[0]).to_value(unit))
            if vmax is None:
                vmax = float(u.Quantity(default[1]).to_value(unit))

            # limits which are far from one have a power of ten factored
            # out of them, into the label of the scale
            exponent = 0
            magnitude = max(abs(vmin), abs(vmax))
            if magnitude > 0 and not 1e-3 <= magnitude < 1e5:
                exponent = int(np.floor(np.log10(magnitude)))

            scales.append((vmin, vmax, unit, exponent))

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
                    moment = _blanked(moment, where)

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
        range_snr: None | tuple[None | float, None | float] = None,
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
        A pixel which is reconstructed with no radiance has a ratio of zero,
        an error of -100%, which is below every bin of a logarithmic scale
        but counts in all that follows.
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

        The pixels of a panel are those within the limits of its horizontal
        scale.
        By default, its vertical scale is symmetric about no error, and
        reaches a tenth beyond the median of the errors of its pixels plus
        three times their scatter, which is half the width of the central
        68% of them.
        The two rows have scales of their own, since the lower row keeps
        pixels which the upper one leaves out.

        Every panel of the upper row is labeled with the Pearson correlation
        coefficient of the true and the reconstructed moment of its pixels,
        the median of their errors, and their scatter.
        For the radiance, the last two are those of the logarithm of the
        ratio, as percentages.

        The signal-to-noise ratio is that of the reconstructed scene, the
        one which an inversion of observed images has, so it rises with the
        reconstructed radiance: of two pixels with the same true radiance,
        the one which is reconstructed brighter has the higher ratio.

        A reconstruction of several frames is compared to the true scene
        frame by frame, and the pixels of every frame are counted together.

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
            upper left panel.
            The scale is in the unit of the first limit which has one, as a
            plain zero has none, and so are those of the other two moments.
            A limit which is :obj:`None` is the 1st or the 99th percentile
            of the pixels in the upper row, since otherwise a few pixels
            would set the scale of all the others, and the lower limit is
            then no further than three decades below the upper one.
            The scale is logarithmic, which has no zero, so a lower limit
            which is not positive is taken as :obj:`None`, as it is for the
            other logarithmic scales.
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
            the left panels, which is logarithmic.
            A limit which is :obj:`None` is that of the default scale of
            each panel.
            Limits which are given are those of both rows, and keep the
            scale the same from one figure to the next, to compare two
            reconstructions.
        range_median_error
            The limits of the error of the median, the vertical scale of the
            middle panels, which is in the unit of the true median,
            whichever unit these limits are in.
            A limit which is :obj:`None` is that of the default scale of
            each panel.
        range_iqr_error
            The limits of the error of the interquartile range, the vertical
            scale of the right panels, which is in the unit of the true
            interquartile range, whichever unit these limits are in.
            A limit which is :obj:`None` is that of the default scale of
            each panel.
        range_snr
            The limits of the signal-to-noise ratio, the horizontal scale of
            the lower row, which is logarithmic.
            A lower limit which is :obj:`None` is one, below which the noise
            is larger than the signal, or the 1st percentile of the ratio if
            that is larger, and an upper limit which is :obj:`None` is its
            99th percentile.
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
            if the limits of a scale do not increase, such as a limit which
            is given beyond the pixels which the other one is taken from,
            or if the instrument cannot be expressed as a pair of sparse
            matrices, which its signal-to-noise ratio is read from.
        """
        instrument = self.inverter.instrument
        solution = self.solution

        self._validate_truth(truth)

        moments_truth = self._moments(truth, axis)
        moments_solution = self._moments(solution, axis)

        snr = instrument.snr(solution)

        # every array is given the axes of all of them, so that the pixels
        # of every frame of a reconstruction are counted
        shape = na.shape_broadcasted(*moments_truth, *moments_solution, snr)
        snr = na.broadcast_to(snr, shape)

        # the pixels of the lower row, and those of the upper row
        radiance_truth = moments_truth[0]
        where_lower = radiance_truth > np.nanpercentile(
            radiance_truth,
            percentile_radiance,
        )
        where_lower = na.broadcast_to(where_lower, shape)
        where_upper = where_lower
        if threshold_snr is not None:
            where_upper = where_lower & (snr > threshold_snr)

        def resolved(
            given: tuple[None | float, None | float],
            default: tuple[float, float],
            name: str,
            log: bool,
        ) -> tuple[float, float]:
            """
            The limits of a scale: those which are given, and the default
            for the others.
            A logarithmic scale has no zero, so a lower limit of one which
            is not positive is not given.
            """
            lower, upper = given
            if lower is None or (log and not lower > 0):
                lower = default[0]
            if upper is None:
                upper = default[1]
            if not lower < upper:
                raise ValueError(
                    f"The limits of `{name}` should increase, but they are "
                    f"{lower:.4g} and {upper:.4g}, where a limit which is "
                    f"not given is that of the pixels."
                )
            return lower, upper

        def statistics(
            error: na.AbstractScalar,
            where: na.AbstractScalar,
            log: bool,
        ) -> tuple[float, float]:
            """
            The median of the errors of the pixels of a panel, and their
            scatter: half the width of the central 68% of them.
            On a logarithmic scale, both are those of the logarithm of the
            errors.
            """
            error = _blanked(error, where)
            if not np.any(np.isfinite(error)).ndarray:
                return np.nan, np.nan

            q = na.ScalarArray(np.array([16, 50, 84]), axes="percentile")
            lower, median, upper = np.nanpercentile(error, q).ndarray

            # the logarithm of a percentile is the percentile of the
            # logarithms, and a ratio of zero has one, unlike its logarithm
            with np.errstate(divide="ignore", invalid="ignore"):
                if log:
                    lower, median, upper = np.log([lower, median, upper])
                scatter = (upper - lower) / 2

            return float(median), float(scatter)

        def symmetric(
            bias: float,
            scatter: float,
            fallback: float,
            log: bool,
        ) -> tuple[float, float]:
            """
            The default limits of a scale of errors: symmetric about no
            error, a tenth beyond their median plus three times their
            scatter.
            If that is no scale, since there are no errors or they are all
            the same but for rounding, the limits are a `fallback` either
            side of no error.
            """
            half = 1.1 * (abs(bias) + 3 * scatter)
            # the exponential of more than this is too large for a float
            ceiling = 700 if log else np.inf
            if not 1e-6 * fallback < half < ceiling:
                half = fallback
            if log:
                return float(np.exp(-half)), float(np.exp(half))
            return -half, half

        def count(
            x: na.AbstractScalar,
            bins_x: na.AbstractScalar,
            where: na.AbstractScalar,
        ) -> na.AbstractScalar:
            """The number of pixels in each column of a histogram."""
            return na.histogram(
                x,
                bins=bins_x,
                weights=where & np.isfinite(x),
            ).outputs

        def histogram(
            x: na.AbstractScalar,
            y: na.AbstractScalar,
            bins_x: na.AbstractScalar,
            bins_y: na.AbstractScalar,
            where: na.AbstractScalar,
            total: na.AbstractScalar,
        ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
            """
            A histogram of `y` against `x`, each column of which is divided
            by the number of its pixels, `total`, whether `y` is within its
            bins or not.
            """
            result = na.histogram2d(
                x,
                y,
                bins=na.Cartesian2dVectorArray(bins_x, bins_y),
                weights=where & np.isfinite(x),
            )

            # a column with no pixels is zero over zero
            with np.errstate(divide="ignore", invalid="ignore"):
                result.outputs = np.nan_to_num(result.outputs / total)

            return result

        def scale_log(scale: matplotlib.axis.Axis) -> None:
            """
            Label a logarithmic scale with plain numbers if it spans less
            than two decades, which powers of ten would label only once or
            twice.
            """
            lower, upper = scale.get_view_interval()
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
                scale.set_major_locator(locator)
                scale.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                scale.set_minor_locator(matplotlib.ticker.NullLocator())

        # the default limits of the horizontal scales are percentiles, since
        # otherwise a few pixels would set the scale of all the others
        percentiles = (1, 99)

        # the signal-to-noise ratio is plotted from one, below which the
        # noise is larger than the signal, unless every ratio is larger
        snr_lower = _blanked(snr, where_lower)
        min_snr = max(1.0, float(_percentile(snr_lower, percentiles[0])))
        max_snr = float(_percentile(snr_lower, percentiles[1]))
        if not min_snr < max_snr:
            # the signal of no pixel is larger than its noise
            max_snr = 10 * min_snr
        limits_snr = resolved(
            given=_limits(range_snr, u.dimensionless_unscaled)[:2],
            default=(min_snr, max_snr),
            name="range_snr",
            log=True,
        )
        bins_snr = na.geomspace(*limits_snr, axis="snr", num=num_bins + 1)
        on_snr = where_lower & (snr >= limits_snr[0]) & (snr <= limits_snr[1])
        total_snr = count(snr, bins_snr, where_lower)

        names = ("radiance", "median", "IQR")
        arguments = ("range_radiance", "range_median", "range_iqr")
        ranges = (range_radiance, range_median, range_iqr)
        ranges_error = (range_radiance_error, range_median_error, range_iqr_error)

        # everything is computed before the figure is made, so that limits
        # which are refused leave no figure behind
        columns = []
        for col, name in enumerate(names):
            # the radiance spans decades and its errors are relative, so it
            # is on a logarithmic scale and its error is a ratio
            log = col == 0
            space = na.geomspace if log else na.linspace

            vmin, vmax, unit = _limits(ranges[col])
            if unit is None:
                unit = na.unit_normalized(moments_truth[col])

            true = na.broadcast_to(_values(moments_truth[col], unit), shape)
            reconstructed = na.broadcast_to(
                _values(moments_solution[col], unit),
                shape,
            )

            # the default limits of the true moment are those of the pixels,
            # unless they all have the same true moment or there are none
            true_upper = _blanked(true, where_upper)
            lower, upper = (float(_percentile(true_upper, q)) for q in percentiles)
            if log:
                lower = max(lower, upper / 1000)
                if not 0 < lower < upper:
                    lower, upper = (lower / 2, 2 * upper) if lower > 0 else (1.0, 10.0)
            elif not lower < upper:
                lower, upper = lower - 0.5, upper + 0.5
            limits_true = resolved((vmin, vmax), (lower, upper), arguments[col], log)

            bins_true = space(*limits_true, axis="true", num=num_bins + 1)
            on_true = where_upper & (true >= limits_true[0]) & (true <= limits_true[1])
            total_true = count(true, bins_true, where_upper)

            if log:
                # a pixel which emits nothing has no ratio, where one which
                # is reconstructed with no radiance has a ratio of zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    error = reconstructed / true
                error = np.where(np.isfinite(error) & (error >= 0), error, np.nan)
                fallback = np.log(2)
                label_error = f"reconstructed / true {name}"
            else:
                error = reconstructed - true
                fallback = (limits_true[1] - limits_true[0]) / 100
                label_error = _label(f"reconstructed $-$ true {name}", unit)

            given_error = _limits(
                ranges_error[col],
                u.dimensionless_unscaled if log else unit,
            )[:2]

            rows = []
            for x, bins_x, where, on, total in (
                (true, bins_true, where_upper, on_true, total_true),
                (snr, bins_snr, where_lower, on_snr, total_snr),
            ):
                bias, scatter = statistics(error, on, log)
                limits_error = resolved(
                    given=given_error,
                    default=symmetric(bias, scatter, fallback, log),
                    name=f"{arguments[col]}_error",
                    log=log,
                )
                bins_error = space(*limits_error, axis="error", num=num_bins + 1)
                hist = histogram(x, error, bins_x, bins_error, where, total)
                rows.append((hist, bias, scatter))

            # the numbers which label the upper panel are those of its pixels
            _, bias, scatter = rows[0]
            if log:
                # adding zero turns a negative zero into a positive one
                text = (
                    f"median error ${np.round(100 * np.expm1(bias)) + 0.0:+.0f}$%\n"
                    f"scatter ${100 * scatter:.0f}$%"
                )
            else:
                suffix = ""
                if unit != u.dimensionless_unscaled:
                    suffix = f" {unit:latex_inline}"
                text = (
                    f"median error ${bias:+.3g}${suffix}\n"
                    f"scatter ${scatter:.3g}${suffix}"
                )

            # the correlation of fewer than two pixels is not defined
            finite = on_true & np.isfinite(true) & np.isfinite(reconstructed)
            r = np.nan
            if finite.sum().ndarray > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = na.stats.pearsonr(true, reconstructed, where=finite)
                r = float(r.ndarray)
            text = f"Pearson's $r = {r:.3f}$\n{text}"

            # where a reconstruction which is the same in every pixel lies
            constant = float(_percentile(_blanked(reconstructed, on_true), 50))

            columns.append((log, unit, label_error, text, constant, rows))

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

        for col, (log, unit, label_error, text, constant, rows) in enumerate(columns):
            for row, (hist, _, _) in enumerate(rows):
                ax = axs[row, col]

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
                log_x = log or row == 1
                if log_x:
                    ax.set_xscale("log")
                if log:
                    ax.set_yscale("log")
                ax.set_xlim(hist.inputs.x.min().ndarray, hist.inputs.x.max().ndarray)
                ax.set_ylim(hist.inputs.y.min().ndarray, hist.inputs.y.max().ndarray)
                if log_x:
                    scale_log(ax.xaxis)
                if log:
                    scale_log(ax.yaxis)
                ax.set_ylabel(label_error)

            ax_true, ax_snr = axs[:, col]

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

            ax_true.text(
                x=0.96,
                y=0.96,
                s=text,
                transform=ax_true.transAxes,
                ha="right",
                va="top",
                fontsize="small",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
            ax_true.set_xlabel(_label(f"true {names[col]}", unit))

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
