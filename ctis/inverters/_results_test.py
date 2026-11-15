import dataclasses
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pytest
import astropy.units as u
import named_arrays as na
import ctis
from ctis.instruments._instruments_test import _instrument_optika, _scene_frames
from ._iterative._mart._mart_test import _IdealInstrumentWeights, instrument, scene

# read noise, so that the signal-to-noise ratio of a pixel depends on it
instrument = dataclasses.replace(instrument, read_noise=2 * u.electron)

inversion = ctis.inverters.MartInverter(
    instrument=instrument,
    threshold_convergence=1e-2,
)(instrument.image(scene))


def _result(
    a: ctis.instruments.AbstractInstrument,
    solution: na.FunctionArray[
        na.AbstractSpectralPositionalVectorArray, na.ScalarArray
    ],
) -> ctis.inverters.InversionResult:
    """
    The result of an inversion with a given instrument, as if the inversion
    had found a given scene.
    """
    return ctis.inverters.InversionResult(
        solution=solution,
        success=True,
        images=a.image(solution, noise=False),
        inverter=ctis.inverters.MartInverter(instrument=a),
        message="",
    )


def test__snr() -> None:
    """
    The signal-to-noise ratio of a pixel of the reconstructed scene is that
    of its voxels alone, summed in quadrature over the channels and the
    wavelengths, as computed from the fields of an ideal instrument.
    """
    result = inversion._snr()

    solution = inversion.solution.outputs

    photons = solution * instrument._volume_scene / instrument._energy_per_photon
    photons = photons * instrument.area_effective * instrument.timedelta_exposure
    photons = photons.to(u.photon).value

    quantum_yield = instrument.quantum_yield.to_value(u.electron / u.photon)
    variance_read = instrument.read_noise.to_value(u.electron) ** 2

    signal = photons * quantum_yield
    variance = signal * quantum_yield + variance_read

    expected = np.square(signal) / variance
    expected = np.sqrt(instrument.num_channel * expected.sum("wavelength"))

    assert result.shape == expected.shape
    assert na.unit(result) is None
    assert result.max() > 5
    assert np.allclose(
        result.ndarray_aligned(expected.axes),
        expected.ndarray,
        rtol=1e-10,
    )


def test__snr_no_operators() -> None:
    """
    The signal-to-noise ratio is computed from the operators of the
    instrument, so an instrument which does not provide them has none, and
    says so.
    """
    a = _IdealInstrumentWeights(
        **{
            f.name: getattr(instrument, f.name)
            for f in dataclasses.fields(instrument)
            if f.init
        }
    )

    with pytest.raises(ValueError, match="pair of sparse matrices"):
        _result(a, inversion.solution)._snr()


def test__snr_empty() -> None:
    """
    A pixel which emits nothing has a signal-to-noise ratio of zero, whether
    the sensor has read noise or not, and computing it does not warn.
    """
    solution = inversion.solution
    index = dict(scene_x=slice(None, 8))
    outputs = solution.outputs.copy()
    outputs[index] = 0
    solution = na.FunctionArray(inputs=solution.inputs, outputs=outputs)

    for read_noise in (0, 2) * u.electron:
        a = dataclasses.replace(instrument, read_noise=read_noise)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = _result(a, solution)._snr()

        assert np.all(result[index] == 0)
        assert np.all(np.isfinite(result))
        assert result.max() > 0


@pytest.mark.parametrize(
    argnames="axis_field",
    argvalues=[
        ("scene_x", "scene_y"),
        ("sensor_x", "sensor_y"),
    ],
    ids=[
        "scene named apart from the pixels",
        "scene named like the pixels",
    ],
)
@pytest.mark.parametrize(
    argnames="energy",
    argvalues=[False, True],
    ids=["scene in photons", "scene in units of energy"],
)
def test__snr_optika(axis_field: tuple[str, str], energy: bool) -> None:
    """
    The signal-to-noise ratio of a pixel is that of the images of that pixel
    alone, for an instrument on a linear system of :mod:`optika`: if the
    sensor has no read noise, the square of the ratio is the signal of the
    images over the variance-to-mean ratio of their shot noise.

    This holds for a scene in units of energy, and for one whose spatial
    axes are named like the pixels of the sensor.
    """
    a = _instrument_optika(axis_field=axis_field)

    # one pixel in the middle of the field, all of whose light reaches
    # the sensor
    index = {ax: 10 for ax in axis_field}
    solution = _scene_frames(a, shape_frames=dict())
    outputs = 0 * solution.outputs
    outputs[index] = 1e3 * solution.outputs[index]
    if energy:
        outputs = (outputs * a._energy_per_photon).to(
            u.erg / u.cm**2 / u.sr / u.nm / u.s
        )
    solution = na.FunctionArray(inputs=solution.inputs, outputs=outputs)

    result = _result(a, solution)._snr()

    images = a.image(solution, noise=False, uncertainty=True).outputs
    signal = images.nominal.to(u.electron).value
    variance = np.square(images.width.to(u.electron).value)
    lit = signal > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.sqrt(np.sum(np.square(signal) / variance, where=lit))

    assert result.shape == {ax: 20 for ax in axis_field}
    assert np.all(result[{ax: slice(11, None) for ax in axis_field}] == 0)
    assert result[index].ndarray > 10
    assert np.isclose(result[index].ndarray, expected.ndarray, rtol=1e-10)

    # the read noise of one pixel for every voxel lowers the ratio
    sensor = dataclasses.replace(a.system.sensor, read_noise=5 * u.electron)
    a = dataclasses.replace(a, system=dataclasses.replace(a.system, sensor=sensor))

    result_read = _result(a, solution)._snr()

    assert 0 < result_read[index].ndarray < result[index].ndarray


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
        num_blank = int((inversion._snr() <= threshold_snr).sum().ndarray)
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
