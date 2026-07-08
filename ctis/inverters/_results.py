import abc
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
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

    def plot_moments(
        self,
        truth: na.FunctionArray[
            na.AbstractDopplerPositionalVectorArray,
            na.ScalarArray,
        ],
        axis: str,
        num_bins: int = 50,
        range_radiance: None | tuple[u.Quantity, u.Quantity] = None,
        range_median: None | tuple[u.Quantity, u.Quantity] = None,
        range_iqr: None | tuple[u.Quantity, u.Quantity] = None,
        percentile_radiance: float = 0,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot column-normalized 2D histograms of the true vs. reconstructed
        radiance, median, and interquartile range of the spectral line profile.

        Parameters
        ----------
        truth
            The true scene which will be compared to the reconstructed scene.
        axis
            The axis along which to compute the radiance, median,
            and interquartile range.
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
        """

        recon = self.solution

        axis_wavelength = self.inverter.instrument.axis_wavelength

        wavelength_truth = truth.inputs.wavelength
        wavelength_recon = recon.inputs.wavelength

        dw_truth = wavelength_truth.volume_cell(axis_wavelength)
        dw_recon = wavelength_recon.volume_cell(axis_wavelength)

        radiance_truth = (truth.outputs * dw_truth).sum(axis_wavelength)
        radiance_recon = (recon.outputs * dw_recon).sum(axis_wavelength)

        thresh = np.nanpercentile(radiance_truth, percentile_radiance)
        where = radiance_truth > thresh

        median_truth = na.pdf.median(
            x=truth.inputs.velocity,
            f=truth.outputs,
            axis=axis,
        )
        median_recon = na.pdf.median(
            x=recon.inputs.velocity,
            f=recon.outputs,
            axis=axis,
        )

        iqr_truth = na.pdf.iqr(
            x=truth.inputs.velocity,
            f=truth.outputs,
            axis=axis,
        )
        iqr_recon = na.pdf.iqr(
            x=recon.inputs.velocity,
            f=recon.outputs,
            axis=axis,
        )

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
