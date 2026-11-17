import dataclasses
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import pytest
import astropy.units as u
import named_arrays as na
import ctis
from ._iterative._mart._mart_test import (
    instrument,
    scene,
    _IdealInstrumentWeights,
)

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
unit_radiance = u.erg / u.s / u.sr / u.cm**2


@pytest.mark.parametrize(
    argnames="kwargs,column,expected",
    argvalues=[
        (dict(range_iqr=(0, 300 * kms)), 2, (0, 300)),
        (dict(range_iqr=(0 * kms, 0.3 * u.Mm / u.s)), 2, (0, 300)),
        (dict(range_iqr=(0.0 * u.Mm / u.s, 300 * kms)), 2, (0, 0.3)),
        (dict(range_median=(-300 * kms, 0)), 1, (-300, 0)),
        (dict(range_iqr=(None, 300 * kms)), 2, (0, 300)),
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
        "no lower limit",
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
    zero, and a limit which is not given is that of the default scale.
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
    reconstructed scene is not compared to it, by the maps or by the
    histograms, and the error names the units of both.
    """
    truth = scene.replace(outputs=scene.outputs / instrument._energy_per_photon)

    for method in (inversion.plot_moments, inversion.plot_moments_histogram):
        with pytest.raises(ValueError, match="true scene is in units of"):
            method(truth)


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


def _histogram(ax: plt.Axes) -> np.ndarray:
    """
    The histogram which is drawn in a panel, with its horizontal axis first.
    """
    mesh = ax.collections[0]
    x = mesh.get_coordinates()[..., 0]
    axis_x = int(x[0, 0] == x[~0, 0])
    return np.moveaxis(np.asarray(mesh.get_array()), axis_x, 0)


def _reconstructed(
    outputs: na.AbstractScalar,
) -> ctis.inverters.AbstractInversionResult:
    """
    The inversion of this module with another reconstructed scene, on the
    same grid.
    """
    solutions = inversion.solutions
    return dataclasses.replace(
        inversion,
        solutions=solutions.replace(outputs=outputs + 0 * solutions.outputs),
    )


@pytest.mark.parametrize("threshold_snr", [None, 5])
@pytest.mark.parametrize("percentile_radiance", [0, 50])
def test_plot_moments_histogram(
    threshold_snr: None | float,
    percentile_radiance: float,
) -> None:
    """
    The histograms of the errors of the moments have a row against the true
    moments and a row against the signal-to-noise ratio, with a column for
    each moment.
    Every column of a histogram is the distribution of the errors of its
    pixels, so it sums to no more than one.
    """
    fig, axs = inversion.plot_moments_histogram(
        truth=scene,
        axis="wavelength",
        percentile_radiance=percentile_radiance,
        threshold_snr=threshold_snr,
    )

    assert isinstance(fig, plt.Figure)
    assert axs.shape == (2, 3)

    for ax in axs.flat:
        assert isinstance(ax, plt.Axes)

        hist = _histogram(ax)

        assert hist.shape == (50, 50)
        assert hist.min() >= 0
        assert hist.max() > 0
        assert hist.sum(1).max() <= 1 + 1e-12

    for ax_true, ax_snr in axs.T:
        labels_true = [line.get_label() for line in ax_true.lines]
        labels_snr = [line.get_label() for line in ax_snr.lines]

        # no error and a constant reconstruction against the true moment,
        # and the threshold against the signal-to-noise ratio
        assert labels_true == ["no error", "constant reconstruction"]
        assert labels_snr[0] == "no error"
        assert ("threshold" in labels_snr) == (threshold_snr is not None)

        assert "Pearson" in ax_true.texts[0].get_text()
        assert ax_snr.get_xscale() == "log"

    # the radiance is on a logarithmic scale, and so is its ratio
    assert axs[0, 0].get_xscale() == "log"
    assert axs[0, 0].get_yscale() == "log"
    assert axs[1, 0].get_yscale() == "log"
    assert axs[0, 1].get_xscale() == "linear"

    plt.close(fig)


@pytest.mark.parametrize("threshold_snr", [None, 5])
@pytest.mark.parametrize("percentile_radiance", [0, 50])
def test_plot_moments_histogram_perfect(
    threshold_snr: None | float,
    percentile_radiance: float,
) -> None:
    """
    A reconstruction which is the true scene has no error in any pixel, so
    every column of a histogram which has pixels has all of them at no
    error.
    The columns of the upper row which have pixels are those of the pixels
    which are neither too faint nor at or below the threshold of the
    signal-to-noise ratio.
    """
    perfect = _reconstructed(scene.outputs)

    fig, axs = perfect.plot_moments_histogram(
        truth=scene,
        range_median=[-300, 300] * kms,
        percentile_radiance=percentile_radiance,
        threshold_snr=threshold_snr,
    )

    for ax in axs.flat:
        hist = _histogram(ax)
        total = hist.sum(1)

        # the two bins on either side of no error
        assert np.all(hist[:, :24] == 0)
        assert np.all(hist[:, 26:] == 0)
        assert np.any(total > 0)
        assert np.all(np.isclose(total, 0) | np.isclose(total, 1))

    for ax in axs[0]:
        assert "$r = 1.000$" in ax.texts[0].get_text()
    assert "median error $+0$%" in axs[0, 0].texts[0].get_text()

    # the pixels of the upper row
    radiance, median, _ = perfect._moments(scene)
    where = radiance > np.nanpercentile(radiance, percentile_radiance)
    if threshold_snr is not None:
        where = where & (instrument.snr(perfect.solution) > threshold_snr)
    expected = na.histogram(
        median,
        bins=na.linspace(-300, 300, axis="true", num=51) * kms,
        weights=where & np.isfinite(median),
    )
    expected = expected.outputs.ndarray

    assert 0 < expected.sum() < median.size
    assert np.array_equal(_histogram(axs[0, 1]).sum(1) > 0, expected > 0)

    plt.close(fig)


def test_plot_moments_histogram_pixels() -> None:
    """
    The numbers which label a panel, and the default limits of its errors,
    are those of the pixels within the limits of its horizontal scale, and
    those limits hold a reconstruction which is biased.
    """
    radiance, _, _ = inversion._moments(scene)
    bright = 100 * unit_radiance

    # a reconstruction which is twice the true scene where that is bright
    result = _reconstructed(np.where(radiance > bright, 2, 1) * scene.outputs)

    for limits, error in [((bright, None), "+100"), ((None, bright), "+0")]:
        fig, axs = result.plot_moments_histogram(scene, range_radiance=limits)

        hist = _histogram(axs[0, 0])
        total = hist.sum(1)

        assert f"median error ${error}$%" in axs[0, 0].texts[0].get_text()
        assert "scatter $0$%" in axs[0, 0].texts[0].get_text()
        assert np.any(total > 0)
        assert np.all(np.isclose(total, 0) | np.isclose(total, 1))

        plt.close(fig)


def test_plot_moments_histogram_biased() -> None:
    """
    The default limits of the errors reach beyond their median, so a
    reconstruction which is three times the true scene is on the scale of
    both rows, which is not a scale of no width although the errors are all
    the same.
    """
    fig, axs = _reconstructed(3 * scene.outputs).plot_moments_histogram(scene)

    for ax in axs[:, 0]:
        lower, upper = ax.get_ylim()
        total = _histogram(ax).sum(1)

        assert lower < 1 / 3
        assert upper > 3
        assert lower == pytest.approx(1 / upper)
        assert np.any(total > 0)
        assert np.all(np.isclose(total, 0) | np.isclose(total, 1))

    assert "median error $+200$%" in axs[0, 0].texts[0].get_text()

    plt.close(fig)


@pytest.mark.parametrize(
    argnames="lower,locator,formatter",
    argvalues=[
        (
            1,
            matplotlib.ticker.LogLocator,
            matplotlib.ticker.LogFormatterSciNotation,
        ),
        (
            10,
            matplotlib.ticker.LogLocator,
            matplotlib.ticker.FormatStrFormatter,
        ),
        (
            40,
            matplotlib.ticker.LogLocator,
            matplotlib.ticker.FormatStrFormatter,
        ),
        (
            100,
            matplotlib.ticker.MaxNLocator,
            matplotlib.ticker.FormatStrFormatter,
        ),
    ],
    ids=[
        "more than two decades",
        "more than a decade",
        "less than a decade",
        "a factor of two",
    ],
)
def test_plot_moments_histogram_ranges(
    lower: float,
    locator: type,
    formatter: type,
) -> None:
    """
    The horizontal scale of each panel has the limits it is given, in the
    unit of those limits, and a limit which is not given is that of the
    pixels.
    The scale of the radiance is logarithmic, and it is labeled with plain
    numbers unless it spans two decades.
    """
    fig, axs = inversion.plot_moments_histogram(
        truth=scene,
        range_radiance=(lower * unit_radiance, 200 * unit_radiance),
        range_median=[-300, 300] * kms,
        range_iqr=(None, 0.4 * u.Mm / u.s),
        range_snr=(2, 20),
    )

    assert axs[0, 0].get_xlim() == pytest.approx((lower, 200))
    assert isinstance(axs[0, 0].xaxis.get_major_locator(), locator)
    assert isinstance(axs[0, 0].xaxis.get_major_formatter(), formatter)

    assert axs[0, 1].get_xlim() == pytest.approx((-300, 300))

    lower_iqr, upper_iqr = axs[0, 2].get_xlim()

    assert upper_iqr == pytest.approx(0.4)
    assert 0 < lower_iqr < upper_iqr

    # the errors are in the units of the limits too
    assert "km" in axs[0, 1].get_ylabel()
    assert "Mm" in axs[0, 2].get_ylabel()

    for ax in axs[1]:
        assert ax.get_xlim() == pytest.approx((2, 20))

    plt.close(fig)


def test_plot_moments_histogram_ranges_error() -> None:
    """
    The vertical scale of each column of panels has the limits it is given,
    converted to the unit of the true moment, and a limit which is not
    given is that of the default scale of each panel, which is symmetric
    about no error.
    """
    fig, axs = inversion.plot_moments_histogram(
        truth=scene,
        range_radiance_error=(0.5, 2),
        range_median_error=[-0.03, 0.03] * u.Mm / u.s,
        range_iqr_error=(0, None),
    )
    fig_default, axs_default = inversion.plot_moments_histogram(truth=scene)

    for row in range(2):
        assert axs[row, 0].get_ylim() == pytest.approx((0.5, 2))
        assert axs[row, 1].get_ylim() == pytest.approx((-30, 30))

        lower, upper = axs[row, 2].get_ylim()
        lower_default, upper_default = axs_default[row, 2].get_ylim()

        assert lower == 0
        assert upper == pytest.approx(upper_default)
        assert lower_default == pytest.approx(-upper_default)

    plt.close(fig)
    plt.close(fig_default)


def test_plot_moments_histogram_zero() -> None:
    """
    A logarithmic scale has no zero, so a lower limit of one which is not
    positive is that of the pixels, as if it were not given.
    The histograms then take the limits of the radiance which the maps take.
    """
    fig, axs = inversion.plot_moments_histogram(
        truth=scene,
        range_radiance=[0, 200] * unit_radiance,
        range_radiance_error=(0, 2),
        range_snr=(0, 20),
    )
    fig_none, axs_none = inversion.plot_moments_histogram(
        truth=scene,
        range_radiance=(None, 200 * unit_radiance),
        range_radiance_error=(None, 2),
        range_snr=(None, 20),
    )

    for ax, ax_none in zip(axs.flat, axs_none.flat):
        assert ax.get_xlim()[0] != 0
        assert ax.get_xlim() == pytest.approx(ax_none.get_xlim())
        assert ax.get_ylim() == pytest.approx(ax_none.get_ylim())

    assert axs[0, 0].get_xlim()[1] == pytest.approx(200)
    assert axs[0, 0].get_ylim()[1] == pytest.approx(2)
    assert axs[1, 0].get_xlim()[1] == pytest.approx(20)

    plt.close(fig)
    plt.close(fig_none)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(range_median=(None, -1000 * kms)),
        dict(range_radiance=(1e6 * unit_radiance, None)),
        dict(range_radiance_error=(None, 0)),
        dict(range_iqr_error=[30, -30] * kms),
        dict(range_snr=(1000, None)),
    ],
    ids=[
        "an upper limit below the pixels",
        "a lower limit above the pixels",
        "an upper limit of zero on a logarithmic scale",
        "limits which decrease",
        "a ratio above that of the pixels",
    ],
)
def test_plot_moments_histogram_order(kwargs: dict) -> None:
    """
    Limits which do not increase are refused before a figure is made, and
    the error names the argument.
    """
    plt.close("all")
    (argument,) = kwargs

    with pytest.raises(ValueError, match=f"`{argument}` should increase"):
        inversion.plot_moments_histogram(scene, **kwargs)

    assert not plt.get_fignums()


def test_plot_moments_histogram_units() -> None:
    """
    A scale is in the unit of the first limit which has one, even if that
    limit is zero, and a limit of the errors which is not a velocity is
    refused before a figure is made.
    """
    fig, axs = inversion.plot_moments_histogram(
        truth=scene,
        range_iqr=(0 * u.Mm / u.s, None),
    )

    assert "Mm" in axs[0, 2].get_xlabel()
    assert "Mm" in axs[0, 2].get_ylabel()
    assert axs[0, 2].get_xlim()[0] == 0

    plt.close("all")

    with pytest.raises(u.UnitConversionError):
        inversion.plot_moments_histogram(scene, range_iqr_error=(0 * u.deg, None))

    assert not plt.get_fignums()


def test_plot_moments_histogram_rows() -> None:
    """
    The lower row keeps the pixels at or below the threshold, so it is the
    same whatever the threshold is, and its default vertical scales are
    those of its own pixels.
    With every pixel left out of the upper row, its histograms are empty.
    """
    fig, axs = inversion.plot_moments_histogram(scene, threshold_snr=5)
    fig_all, axs_all = inversion.plot_moments_histogram(scene, threshold_snr=np.inf)

    for ax, ax_all in zip(axs[1], axs_all[1]):
        assert ax.get_xlim() == pytest.approx(ax_all.get_xlim())
        assert ax.get_ylim() == pytest.approx(ax_all.get_ylim())
        assert np.allclose(_histogram(ax), _histogram(ax_all))

    # the errors of the median are larger below the threshold than above it
    assert axs[1, 1].get_ylim()[1] > axs[0, 1].get_ylim()[1]

    for ax in axs_all[0]:
        assert _histogram(ax).max() == 0
        assert np.all(np.isfinite(ax.get_xlim()))
        assert np.all(np.isfinite(ax.get_ylim()))

    plt.close(fig)
    plt.close(fig_all)


def test_plot_moments_histogram_frames() -> None:
    """
    The pixels of every frame of a reconstruction are counted, so the same
    reconstruction in each of three frames has the histograms of one frame,
    and its correlation with the true scene.
    """
    kwargs = dict(
        truth=scene,
        range_radiance=(1 * unit_radiance, 200 * unit_radiance),
        range_median=[-300, 300] * kms,
        range_iqr=[0, 400] * kms,
        range_radiance_error=(0.5, 2),
        range_median_error=[-30, 30] * kms,
        range_iqr_error=[-30, 30] * kms,
        range_snr=(1, 20),
    )
    frames = na.ScalarArray(np.ones(3), axes="time")
    result = _reconstructed(inversion.solutions.outputs * frames)

    fig, axs = result.plot_moments_histogram(**kwargs)
    fig_one, axs_one = inversion.plot_moments_histogram(**kwargs)

    for ax, ax_one in zip(axs.flat, axs_one.flat):
        hist = _histogram(ax)

        assert hist.max() > 0
        assert hist.sum(1).max() <= 1 + 1e-12
        assert np.allclose(hist, _histogram(ax_one))

    # the median and the scatter are percentiles, which are interpolated
    # between other pixels in three copies of them than in one, so only the
    # correlation, the first line of a label, is the same to its last digit
    for ax, ax_one in zip(axs[0], axs_one[0]):
        label = ax.texts[0].get_text().splitlines()
        label_one = ax_one.texts[0].get_text().splitlines()

        assert label[0] == label_one[0]
        assert len(label) == len(label_one) == 3

    plt.close(fig)
    plt.close(fig_one)


def test_plot_moments_histogram_no_radiance() -> None:
    """
    A pixel which is reconstructed with no radiance has an error of -100%,
    which is below every bin of a logarithmic scale, but it counts in its
    column and in the median of the errors.
    """
    shape = scene.outputs.shape
    index_x = na.ScalarArray(np.arange(shape["scene_x"]), axes="scene_x")
    index_y = na.ScalarArray(np.arange(shape["scene_y"]), axes="scene_y")

    # the true scene in a third of the pixels, and nothing in the others
    result = _reconstructed(((index_x + index_y) % 3 == 0) * scene.outputs)

    fig, axs = result.plot_moments_histogram(scene)

    total = _histogram(axs[0, 0]).sum(1)

    assert "median error $-100$%" in axs[0, 0].texts[0].get_text()
    assert np.any(total > 0)
    assert np.all(total < 0.9)

    plt.close(fig)


def test_plot_moments_histogram_dark() -> None:
    """
    A reconstruction which emits nothing has an error of -100% in its
    radiance, and neither a median nor an interquartile range, so its
    histograms are empty, and they are plotted without a warning, on scales
    whose limits are numbers.
    """
    dark = _reconstructed(0 * inversion.solutions.outputs)

    for threshold_snr in (None, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, axs = dark.plot_moments_histogram(
                truth=scene,
                threshold_snr=threshold_snr,
            )

        for ax in axs.flat:
            assert _histogram(ax).max() == 0
            assert np.all(np.isfinite(ax.get_xlim()))
            assert np.all(np.isfinite(ax.get_ylim()))

        for ax in axs[0]:
            assert "nan" in ax.texts[0].get_text()

        plt.close(fig)


def test_plot_moments_histogram_unsupported() -> None:
    """
    The lower row is against the signal-to-noise ratio, which is read from
    the sparse matrices of the instrument, so the histograms of an
    instrument which does not provide them are not plotted, and the error
    says why.
    """
    unsupported = _IdealInstrumentWeights(
        **{
            field.name: getattr(instrument, field.name)
            for field in dataclasses.fields(instrument)
            if field.init
        }
    )
    result = dataclasses.replace(
        inversion,
        inverter=dataclasses.replace(inversion.inverter, instrument=unsupported),
    )

    with pytest.raises(ValueError, match="pair of sparse matrices"):
        result.plot_moments_histogram(scene)
