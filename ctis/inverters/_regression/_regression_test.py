import warnings
import pytest
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import ctis
from .._iterative._iterative_test import AbstractTestAbstractIterativeInverter

torch = pytest.importorskip("torch")


wavelength_rest = 630 * u.AA


def _instrument(
    num_velocity: int,
    num_scene: int,
    num_sensor: int,
    num_channel: int,
    dispersion: u.Quantity,
) -> ctis.instruments.IdealInstrument:
    velocity = na.linspace(-200, 200, axis="wavelength", num=num_velocity + 1)
    velocity = velocity * u.km / u.s
    coordinates_scene = na.DopplerPositionalVectorArray.from_velocity(
        velocity=velocity,
        wavelength_rest=wavelength_rest,
        position=na.Cartesian2dVectorLinearSpace(
            start=-4 * u.arcsec,
            stop=+4 * u.arcsec,
            axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
            num=num_scene + 1,
        ),
    )
    coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
        velocity=velocity,
        wavelength_rest=wavelength_rest,
        position=na.Cartesian2dVectorArray(
            x=na.arange(0, num_sensor + 1, axis="sensor_x") * u.pix,
            y=na.arange(0, num_sensor + 1, axis="sensor_y") * u.pix,
        ),
    )
    angle = na.linspace(0, 360, num=num_channel, axis="channel", endpoint=False)
    angle = angle * u.deg
    return ctis.instruments.IdealInstrument(
        area_effective=1 * u.cm**2,
        timedelta_exposure=20 * u.s,
        plate_scale=8 / num_scene * u.arcsec / u.pix,
        dispersion=dispersion,
        angle=angle,
        wavelength_ref=wavelength_rest,
        position_ref=num_sensor / 2 * u.pix,
        coordinates_scene=coordinates_scene,
        coordinates_sensor=coordinates_sensor,
        channel=angle,
        axis_channel="channel",
        axis_wavelength="wavelength",
        axis_scene_xy=("scene_x", "scene_y"),
        axis_sensor_xy=("sensor_x", "sensor_y"),
    )


#: an underdetermined instrument, with more voxels than pixels
instrument = _instrument(
    num_velocity=12,
    num_scene=16,
    num_sensor=24,
    num_channel=3,
    dispersion=0.105 * u.AA / u.pix,
)

axes_scene = ("wavelength", "scene_x", "scene_y")
axes_images = ("channel", "sensor_x", "sensor_y")

scene = ctis.scenes.gaussians(instrument.coordinates_scene)
scene = scene.replace(outputs=scene.outputs.to(u.erg / (u.cm**2 * u.sr * u.s * u.AA)))
images = instrument.image(scene, noise=False)
images_uncertain = instrument.image(scene, noise=False, uncertainty=True)
images_noisy = images.replace(
    outputs=na.ScalarArray(
        ndarray=np.random.default_rng(0).poisson(images.outputs.ndarray.value)
        << u.electron,
        axes=images.outputs.axes,
    ),
)

unit = ctis.inverters.ElasticNetInverter(instrument=instrument).unit_scene

prior = scene.outputs.mean(("scene_x", "scene_y"))
penalty_weights = na.ScalarArray(
    ndarray=1 + np.arange(16 * 16).reshape(16, 16) / 100,
    axes=("scene_x", "scene_y"),
)

devices = ["cpu"]
if torch.cuda.is_available():  # pragma: nocover
    devices.append("cuda")


class AbstractTestAbstractRegressionInverter(
    AbstractTestAbstractIterativeInverter,
):

    def test_alpha(self, a: ctis.inverters.AbstractRegressionInverter):
        assert a.alpha >= 0

    def test_l1_ratio(self, a: ctis.inverters.AbstractRegressionInverter):
        assert 0 <= a.l1_ratio <= 1


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        ctis.inverters.ElasticNetInverter(
            instrument=instrument,
            device=device,
        )
        for device in devices
    ]
    + [
        ctis.inverters.ElasticNetInverter(
            instrument=instrument,
            alpha=1e-3,
            l1_ratio=0.2,
            prior=prior,
            penalty_weights=penalty_weights,
            device="cpu",
        ),
        ctis.inverters.ElasticNetInverter(
            instrument=instrument,
            num_iteration=25,
            intermediate=True,
            device="cpu",
        ),
        ctis.inverters.ElasticNetInverter(
            instrument=instrument,
            alpha=0,
            device="cpu",
        ),
        ctis.inverters.RidgeInverter(
            instrument=instrument,
            device="cpu",
        ),
        ctis.inverters.LassoInverter(
            instrument=instrument,
            device="cpu",
        ),
    ],
)
class TestElasticNetInverter(
    AbstractTestAbstractRegressionInverter,
):

    @pytest.mark.parametrize("images", [images, images_uncertain])
    @pytest.mark.parametrize("guess", [None, scene])
    def test__call__(
        self,
        a: ctis.inverters.ElasticNetInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        guess: None | na.AbstractFunctionArray,
    ):
        # whether the iteration converges within its cap depends on the
        # configuration, so the warning is neither required nor forbidden here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = super().test__call__(a=a, images=images, guess=guess)

        assert isinstance(result, ctis.inverters.RegressionInversionResult)

        axis_iteration = a.axis_iteration

        solution = result.solution.outputs
        assert np.all(solution >= 0)
        assert solution.unit.is_equivalent(a.unit_scene)

        assert result.solutions.outputs.shape[axis_iteration] == (
            result.num_iteration if a.intermediate else 1
        )

        # the merit function never increases
        objective = result.objective.ndarray
        assert objective.size == result.num_iteration
        assert np.all(np.diff(objective) <= 1e-5 * objective[:-1])

        assert result.scale > 0
        assert result.duality_gap >= -1e-6

    def test_unit_scene(self, a: ctis.inverters.ElasticNetInverter):
        assert a.unit_scene.is_equivalent(u.erg / (u.cm**2 * u.sr * u.s * u.AA))


#: an instrument small enough that the forward model can be written as a
#: dense matrix
instrument_small = _instrument(
    num_velocity=4,
    num_scene=6,
    num_sensor=12,
    num_channel=3,
    dispersion=0.2 * u.AA / u.pix,
)


def _problem_small():
    """
    A random sparse scene observed by the small instrument,
    and the forward model as a dense matrix.
    """
    inverter = ctis.inverters.ElasticNetInverter(
        instrument=instrument_small,
        device="cpu",
        dtype=torch.float64,
    )
    regridder = inverter._regridder()
    forward, transpose = inverter._operator(regridder)

    shape = tuple(regridder.shape_input[ax] for ax in axes_scene)
    num = int(np.prod(shape))

    rng = np.random.default_rng(1)
    truth = np.where(rng.random(shape) < 0.3, rng.random(shape), 0) * 3e3

    image = forward(torch.as_tensor(truth)).numpy()
    data = rng.poisson(image).astype(float)
    sigma = np.sqrt(np.maximum(data, 1) + np.square(4.0))

    matrix = np.stack(
        arrays=[
            forward(torch.as_tensor(np.eye(num)[j].reshape(shape))).numpy().ravel()
            for j in range(num)
        ],
        axis=1,
    )

    images = na.FunctionArray(
        inputs=instrument_small.coordinates_sensor,
        outputs=na.ScalarArray(data << u.electron, axes=axes_images),
    )
    uncertainty = na.ScalarArray(sigma << u.electron, axes=axes_images)

    return images, uncertainty, matrix, shape, forward, transpose


@pytest.mark.parametrize(
    argnames="alpha,l1_ratio,has_prior,has_weights",
    argvalues=[
        (1, 0.5, False, False),
        (10, 0.0, True, False),
        (1, 0.7, True, True),
        (0.3, 0.5, False, True),
    ],
)
def test__call__nnls(
    alpha: float,
    l1_ratio: float,
    has_prior: bool,
    has_weights: bool,
):
    r"""
    A nonnegative elastic net is a nonnegative least-squares problem in
    disguise: the :math:`\ell_2` penalty appends one row per voxel to the
    matrix, and the :math:`\ell_1` penalty shifts the prior of those rows.
    The reconstruction must match an independent solution of that problem.
    """
    images, uncertainty, matrix, shape, _, _ = _problem_small()
    num_sample, num_voxel = matrix.shape

    rng = np.random.default_rng(2)
    x0 = rng.random(shape) * 2e3 if has_prior else np.zeros(shape)
    c = 0.2 + 2 * rng.random(shape) if has_weights else np.ones(shape)

    inverter = ctis.inverters.ElasticNetInverter(
        instrument=instrument_small,
        alpha=alpha,
        l1_ratio=l1_ratio,
        prior=na.ScalarArray(x0 << unit, axes=axes_scene) if has_prior else None,
        penalty_weights=na.ScalarArray(c, axes=axes_scene) if has_weights else None,
        uncertainty=uncertainty,
        threshold_convergence=1e-10,
        num_iteration=20000,
        device="cpu",
        dtype=torch.float64,
    )
    result = inverter(images)

    assert result.success
    assert result.duality_gap <= 1e-10

    scale = result.scale.to_value(unit)
    data = images.outputs.ndarray.to_value(u.electron).ravel()
    w = 1 / np.square(uncertainty.ndarray.to_value(u.electron).ravel())

    l1 = alpha * l1_ratio / num_voxel
    l2 = alpha * (1 - l1_ratio) / num_voxel
    shift = x0.ravel() / scale - l1 * c.ravel() / l2

    matrix_augmented = np.concatenate(
        [
            np.sqrt(w / num_sample)[:, np.newaxis] * matrix * scale,
            np.sqrt(l2) * np.eye(num_voxel),
        ]
    )
    data_augmented = np.concatenate(
        [
            np.sqrt(w / num_sample) * data,
            np.sqrt(l2) * shift,
        ]
    )
    expected, _ = scipy.optimize.nnls(matrix_augmented, data_augmented)
    expected = expected * scale

    solution = result.solution.outputs.ndarray.to_value(unit).ravel()

    assert np.allclose(solution, expected, atol=1e-4 * expected.max())

    # the scene is sparse, and the constraint finds the empty voxels exactly
    assert np.any(solution == 0)


def test__call__lasso():
    """
    Without an :math:`\\ell_2` penalty there is no closed form to compare to,
    so check the optimality conditions directly: the gradient of the merit
    function must vanish in every voxel which is filled, and must not be
    negative in any voxel which is empty.
    """
    images, uncertainty, matrix, shape, _, _ = _problem_small()
    num_sample, num_voxel = matrix.shape

    alpha = 1

    inverter = ctis.inverters.LassoInverter(
        instrument=instrument_small,
        alpha=alpha,
        uncertainty=uncertainty,
        threshold_convergence=1e-10,
        num_iteration=20000,
        device="cpu",
        dtype=torch.float64,
    )
    result = inverter(images)

    assert result.success

    scale = result.scale.to_value(unit)
    data = images.outputs.ndarray.to_value(u.electron).ravel()
    w = 1 / np.square(uncertainty.ndarray.to_value(u.electron).ravel())

    solution = result.solution.outputs.ndarray.to_value(unit).ravel() / scale

    matrix = matrix * scale
    gradient = matrix.T @ (w * (matrix @ solution - data)) / num_sample
    gradient = gradient + alpha / num_voxel

    filled = solution > 0
    tolerance = 1e-4 * np.abs(gradient).max()

    assert np.any(filled)
    assert np.any(~filled)
    assert np.all(np.abs(gradient[filled]) < tolerance)
    assert np.all(gradient[~filled] > -tolerance)


def test__call__guess():
    """
    The merit function has a single minimum,
    so the reconstruction must not depend on where the iteration starts.
    """
    inverter = ctis.inverters.ElasticNetInverter(
        instrument=instrument,
        alpha=1,
        threshold_convergence=1e-8,
        num_iteration=20000,
        device="cpu",
        dtype=torch.float64,
    )

    result_a = inverter(images_noisy)
    result_b = inverter(images_noisy, guess=10 * scene)

    assert result_a.success
    assert result_b.success

    solution_a = result_a.solution.outputs.ndarray
    solution_b = result_b.solution.outputs.ndarray

    assert np.abs(solution_a - solution_b).max() < 1e-3 * solution_a.max()


def test__call__prior():
    """
    As the penalty becomes very strong, ridge regression must return the prior.
    """
    inverter = ctis.inverters.RidgeInverter(
        instrument=instrument,
        alpha=1e12,
        prior=scene,
        device="cpu",
        dtype=torch.float64,
    )
    result = inverter(images)

    assert np.allclose(result.solution.outputs, scene.outputs)


def test__call__penalty_weights():
    """
    A large weight on some of the voxels must push the radiance out of them.
    """
    kwargs = dict(
        instrument=instrument,
        alpha=1e-1,
        l1_ratio=0.9,
        device="cpu",
    )

    where = instrument.coordinates_scene.position.x.cell_centers() > 0 * u.arcsec
    weights = np.where(where, 1e6, 1)

    result = ctis.inverters.ElasticNetInverter(**kwargs)(images)
    result_weighted = ctis.inverters.ElasticNetInverter(
        penalty_weights=weights,
        **kwargs,
    )(images)

    total = result.solution.outputs.sum(where=where)
    total_weighted = result_weighted.solution.outputs.sum(where=where)

    assert total > 0
    assert total_weighted < 1e-3 * total


def test__call__uncertainty():
    """
    For an ideal instrument without read noise, the default uncertainty is the
    shot noise of the measured signal, floored at one photon.
    """
    kwargs = dict(
        instrument=instrument,
        num_iteration=50,
        device="cpu",
        dtype=torch.float64,
    )

    width = np.sqrt(np.maximum(images.outputs.to_value(u.electron), 1)) * u.electron

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = ctis.inverters.ElasticNetInverter(**kwargs)(images)
        result_explicit = ctis.inverters.ElasticNetInverter(
            uncertainty=width,
            **kwargs,
        )(images)

    assert np.allclose(result.solution.outputs, result_explicit.solution.outputs)

    # a pixel whose variance is zero is ignored instead of dividing by zero
    inverter_zero = ctis.inverters.ElasticNetInverter(
        instrument=instrument,
        uncertainty=np.where(images.outputs > 0 * u.electron, width, 0 * u.electron),
        num_iteration=20,
        device="cpu",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_zero = inverter_zero(images)

    assert np.all(np.isfinite(result_zero.solution.outputs))


@pytest.mark.parametrize(
    argnames="inverter",
    argvalues=[
        ctis.inverters.ElasticNetInverter(instrument=instrument, device="cpu"),
        ctis.inverters.LassoInverter(instrument=instrument, device="cpu"),
    ],
)
def test__call__empty(inverter: ctis.inverters.ElasticNetInverter):
    """An observation of nothing must be reconstructed as nothing."""
    result = inverter(images.replace(outputs=0 * images.outputs))

    assert result.success
    assert np.all(result.solution.outputs == 0)


def test__call__unweighted():
    """If every pixel is ignored, only the penalty remains."""
    inverter = ctis.inverters.ElasticNetInverter(
        instrument=instrument,
        uncertainty=0 * u.electron,
        device="cpu",
    )
    result = inverter(images)

    assert result.success
    assert np.all(result.solution.outputs == 0)


def test__call__recovery():
    """
    A noiseless observation must be reproduced by the reconstruction.
    """
    inverter = ctis.inverters.ElasticNetInverter(
        instrument=instrument,
        alpha=1e-4,
        device="cpu",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = inverter(images)

    assert np.all(result.mean_chi_squared[{inverter.axis_iteration: ~0}] < 1)

    axis = "wavelength"
    r = na.stats.pearsonr(
        scene.outputs.sum(axis),
        result.solution.outputs.sum(axis),
        axis=("scene_x", "scene_y"),
    )
    assert r > 0.9


def test__call__unit():
    unit_new = u.W / (u.m**2 * u.sr * u.nm)
    kwargs = dict(
        instrument=instrument,
        num_iteration=20,
        device="cpu",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = ctis.inverters.ElasticNetInverter(**kwargs)(images)
        result_new = ctis.inverters.ElasticNetInverter(unit=unit_new, **kwargs)(images)

    solution = result.solution.outputs.ndarray
    solution_new = result_new.solution.outputs.ndarray

    assert solution_new.unit == unit_new
    assert np.abs(solution_new - solution).max() < 1e-3 * solution.max()


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(alpha=-1),
        dict(l1_ratio=-0.1),
        dict(l1_ratio=1.1),
    ],
)
def test__post_init__invalid(kwargs: dict):
    with pytest.raises(ValueError):
        ctis.inverters.ElasticNetInverter(instrument=instrument, **kwargs)


def test__post_init__invalid_instrument():
    with pytest.raises(ValueError, match="AbstractLinearInstrument"):
        ctis.inverters.ElasticNetInverter(instrument=None)


def test__call__invalid():
    inverter = ctis.inverters.ElasticNetInverter(instrument=instrument, device="cpu")

    images_shifted = images.replace(
        inputs=images.inputs.replace(position=images.inputs.position + 1 * u.pix),
    )
    with pytest.raises(ValueError, match="are not equal"):
        inverter(images_shifted)

    images_stacked = images.replace(
        outputs=na.stack([images.outputs, images.outputs], axis="time"),
    )
    with pytest.raises(ValueError, match="not one of the axes"):
        inverter(images_stacked)

    inverter_negative = ctis.inverters.ElasticNetInverter(
        instrument=instrument,
        penalty_weights=-1,
        device="cpu",
    )
    with pytest.raises(ValueError, match="nonnegative"):
        inverter_negative(images)
