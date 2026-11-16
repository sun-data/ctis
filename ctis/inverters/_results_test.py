import dataclasses
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pytest
import astropy.units as u
import named_arrays as na
import ctis
from ._iterative._mart._mart_test import instrument, scene

# read noise, so that the signal-to-noise ratio of a pixel depends on it
instrument = dataclasses.replace(instrument, read_noise=2 * u.electron)

inversion = ctis.inverters.MartInverter(
    instrument=instrument,
    threshold_convergence=1e-2,
)(instrument.image(scene))


@pytest.mark.parametrize("truth", [None, scene])
@pytest.mark.parametrize("threshold_snr", [None, 5])
def test_plot_moments(
    truth: None | na.FunctionArray,
    threshold_snr: None | float,
) -> None:
    """
    The maps of the moments have a row for each scene and a column for each
    moment, and the median and the interquartile range are blank wherever
    the signal-to-noise ratio is at or below the threshold.
    """
    fig, axs = inversion.plot_moments(
        axis="wavelength",
        truth=truth,
        threshold_snr=threshold_snr,
    )

    assert isinstance(fig, plt.Figure)
    assert axs.shape == (1 if truth is None else 2, 3)
    for ax in axs.flat:
        assert isinstance(ax, plt.Axes)

    num_blank = 0
    if threshold_snr is not None:
        snr = instrument.snr(inversion.solution)
        num_blank = int((snr <= threshold_snr).sum().ndarray)
        assert num_blank > 0

    for ax_radiance, ax_median, ax_iqr in axs:
        radiance = ax_radiance.collections[0].get_array()
        median = ax_median.collections[0].get_array()
        iqr = ax_iqr.collections[0].get_array()

        assert np.ma.count_masked(radiance) == 0
        assert np.ma.count_masked(median) >= num_blank
        assert np.ma.count_masked(iqr) >= num_blank
        assert np.ma.count(median) > 0
        assert np.ma.count(iqr) > 0

    plt.close(fig)


def test_plot_moments_ranges() -> None:
    """
    The color scale of each moment has the limits it is given, in the unit
    of those limits, and limits which are far from one are factored out of
    the scale, into its label.
    """
    fig, axs = inversion.plot_moments(
        axis="wavelength",
        range_radiance=[0, 200] * u.erg / u.s / u.sr / u.cm**2,
        range_median=[-300, 300] * u.km / u.s,
        range_iqr=[0, 0.4] * u.Mm / u.s,
    )

    assert axs[0, 0].collections[0].get_clim() == (0, 200)
    assert axs[0, 1].collections[0].get_clim() == (-300, 300)
    assert axs[0, 2].collections[0].get_clim() == (0, 0.4)

    plt.close(fig)

    # the reconstructed scene is in the units of the backprojection, in
    # which its radiance is a small number
    fig, axs = inversion.plot_moments(axis="wavelength")

    vmin, vmax = axs[0, 0].collections[0].get_clim()
    label = fig.axes[axs.size].get_xlabel()

    assert vmin == 0
    assert 1 <= vmax < 10
    assert "$10^{" in label

    plt.close(fig)


def test_plot_moments_axis() -> None:
    """
    The median and the interquartile range are computed along the wavelength
    axis of the instrument unless another axis is given, by both methods,
    which take the true scene first.
    """
    axis = instrument.axis_wavelength

    for method, args in [
        (inversion.plot_moments, ()),
        (inversion.plot_moments, (scene,)),
        (inversion.plot_moments_histogram, (scene,)),
    ]:
        fig, axs = method(*args)
        fig_axis, axs_axis = method(*args, axis=axis)

        assert axs.shape == axs_axis.shape
        for ax, ax_axis in zip(axs.flat, axs_axis.flat):
            assert np.ma.allequal(
                ax.collections[0].get_array(),
                ax_axis.collections[0].get_array(),
            )

        plt.close(fig)
        plt.close(fig_axis)


kms = u.km / u.s


@pytest.mark.parametrize(
    argnames="kwargs,column,expected",
    argvalues=[
        (dict(range_iqr=(0, 300 * kms)), 2, (0, 300)),
        (dict(range_iqr=(0 * kms, 0.3 * u.Mm / u.s)), 2, (0, 300)),
        (dict(range_iqr=(0.0 * u.Mm / u.s, 300 * kms)), 2, (0, 0.3)),
        (dict(range_median=(-300 * kms, 0)), 1, (-300, 0)),
        (
            dict(range_median=(na.ScalarArray(-300 * kms), na.ScalarArray(300 * kms))),
            1,
            (-300, 300),
        ),
    ],
    ids=[
        "plain zero below",
        "two units",
        "zero with a unit",
        "plain zero above",
        "named arrays",
    ],
)
def test_plot_moments_limits(
    kwargs: dict,
    column: int,
    expected: tuple[float, float],
) -> None:
    """
    The limits of a color scale can be quantities or named arrays, in units
    of their own, and a plain zero beside either.
    The scale is in the unit of its lower limit, unless that is a plain
    zero.
    """
    fig, axs = inversion.plot_moments(**kwargs)

    assert axs[0, column].collections[0].get_clim() == pytest.approx(expected)

    plt.close(fig)


def test_plot_moments_dark() -> None:
    """
    A reconstruction which emits nothing has neither a median nor an
    interquartile range anywhere, and its maps are plotted without a
    warning, on scales whose limits are numbers.
    """
    solutions = inversion.solutions
    dark = dataclasses.replace(
        inversion,
        solutions=solutions.replace(outputs=0 * solutions.outputs),
    )

    for threshold_snr in (None, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, axs = dark.plot_moments(threshold_snr=threshold_snr)

        for ax in axs.flat:
            vmin, vmax = ax.collections[0].get_clim()
            assert np.isfinite(vmin) and np.isfinite(vmax)

        plt.close(fig)


def test_plot_moments_frames() -> None:
    """
    The maps are of one scene, so the reconstruction of several frames is
    not plotted, and the error says why.
    """
    solutions = inversion.solutions
    frames = na.ScalarArray(np.array([1, 2]), axes="time")
    result = dataclasses.replace(
        inversion,
        solutions=solutions.replace(outputs=solutions.outputs * frames),
    )

    with pytest.raises(ValueError, match="maps of one scene"):
        result.plot_moments()


def test_plot_moments_truth_unit() -> None:
    """
    A true scene in units which cannot be converted to those of the
    reconstructed scene is not plotted on its color scales, and the error
    names the units of both.
    """
    truth = scene.replace(outputs=scene.outputs / instrument._energy_per_photon)

    with pytest.raises(ValueError, match="true scene is in units of"):
        inversion.plot_moments(truth)


@pytest.mark.parametrize("threshold_snr", [5, np.inf])
def test_plot_moments_no_runtime_warnings(threshold_snr: float) -> None:
    """
    The pixels of the scene which emit nothing have neither a median nor a
    signal-to-noise ratio, so plotting the maps of the moments must not warn
    about them, nor about a threshold which leaves every pixel blank.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig, axs = inversion.plot_moments(
            axis="wavelength",
            truth=scene,
            threshold_snr=threshold_snr,
        )

    for ax in axs[:, 1:].flat:
        vmin, vmax = ax.collections[0].get_clim()
        assert np.isfinite(vmin) and np.isfinite(vmax)

    plt.close(fig)


@pytest.mark.parametrize("threshold_snr", [None, 5])
def test_plot_moments_histogram(threshold_snr: None | float) -> None:
    """
    The histograms of the moments leave out the pixels whose signal-to-noise
    ratio is at or below the threshold.
    """
    fig, axs = inversion.plot_moments_histogram(
        truth=scene,
        axis="wavelength",
        threshold_snr=threshold_snr,
    )

    assert isinstance(fig, plt.Figure)
    assert len(axs) == 3
    for ax in axs:
        assert isinstance(ax, plt.Axes)

    plt.close(fig)
