from typing import Callable
import pytest
import numpy as np
import named_arrays as na
import ctis
from . import _kernels

num_channel = 3
num_pixel = 50
num_voxel = 40


def _images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Observed images, predicted images, and the variance of the predicted
    images, which is zero in the first pixels of each channel.
    """
    rng = np.random.default_rng(0)
    predicted = rng.uniform(50, 150, size=(num_channel, num_pixel))
    observed = rng.poisson(predicted).astype(float)
    variance = predicted.copy()
    variance[:, :5] = 0
    return observed, predicted, variance


@pytest.mark.parametrize(
    argnames="merit",
    argvalues=[_kernels.merit, _kernels.merit.py_func],
    ids=["compiled", "python"],
)
class TestMerit:

    def test_merit(self, merit: Callable) -> None:
        """
        The kernel gives the same mean chi squared and correlation as the
        merit functions of this package give for named arrays.
        """
        observed, predicted, variance = _images()
        chi_squared = np.empty(num_channel)
        correlation = np.empty(num_channel)

        merit(observed, predicted, variance, chi_squared, correlation)

        axes = ("channel", "pixel")
        observed = na.ScalarArray(observed, axes=axes)
        predicted = na.ScalarArray(predicted, axes=axes)
        uncertainty = na.ScalarArray(np.sqrt(variance), axes=axes)
        chi_squared_expected = ctis.inverters.merit.mean_chi_squared(
            observed=observed,
            expected=predicted,
            uncertainty=uncertainty,
            axis="pixel",
        )
        correlation_expected = ctis.inverters.merit.correlation_residual(
            observed=observed,
            expected=predicted,
            axis="pixel",
        )

        assert np.allclose(chi_squared, chi_squared_expected.ndarray, rtol=1e-12)
        assert np.allclose(correlation, correlation_expected.ndarray, atol=1e-12)
        assert np.all(chi_squared > 0)
        assert np.all(np.abs(correlation) < 1)

    def test_merit_empty(self, merit: Callable) -> None:
        """
        A channel without noise anywhere has no mean chi squared, and a
        channel which is the same everywhere has no correlation.
        """
        observed, predicted, variance = _images()
        variance[1] = 0
        predicted[2] = 100
        chi_squared = np.empty(num_channel)
        correlation = np.empty(num_channel)

        merit(observed, predicted, variance, chi_squared, correlation)

        assert np.array_equal(np.isnan(chi_squared), [False, True, False])
        assert np.array_equal(np.isnan(correlation), [False, False, True])


def _update(
    scene: np.ndarray,
    backprojected: np.ndarray,
    backprojected_new: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """The update of the scene as the loop on named arrays computes it."""
    backprojected = np.maximum(backprojected, 0)
    backprojected_new = np.maximum(backprojected_new, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = backprojected / backprojected_new
    correction = np.nan_to_num(correction, nan=1, posinf=1, neginf=1)
    correction = correction**gamma
    correction = np.prod(correction, axis=0)
    correction = correction ** (1 / backprojected.shape[0])
    return scene * correction


@pytest.mark.parametrize(
    argnames="update",
    argvalues=[_kernels.update, _kernels.update.py_func],
    ids=["compiled", "python"],
)
class TestUpdate:

    @pytest.mark.parametrize("gamma", [1, 0.5])
    def test_update(self, update: Callable, gamma: float) -> None:
        rng = np.random.default_rng(0)
        scene = rng.uniform(1, 2, size=num_voxel)
        backprojected = rng.uniform(1, 2, size=(num_channel, num_voxel))
        backprojected_new = rng.uniform(1, 2, size=(num_channel, num_voxel))

        expected = _update(scene, backprojected, backprojected_new, gamma)

        update(scene, backprojected, backprojected_new, gamma)

        assert np.allclose(scene, expected, rtol=1e-12)
        assert not np.allclose(scene, 1)

    def test_update_undefined(self, update: Callable) -> None:
        """
        A correction which is not finite is one, a negative backprojection is
        zero, and a voxel with no signal in a channel is zeroed.
        """
        special = [-1.0, -0.0, 0.0, 1.0, 2.0, np.inf, -np.inf, np.nan]
        numerator, denominator = np.meshgrid(special, special, indexing="ij")
        numerator = numerator.reshape(-1)
        denominator = denominator.reshape(-1)

        rng = np.random.default_rng(0)
        scene = rng.uniform(1, 2, size=numerator.size)
        backprojected = np.stack([numerator, np.full(numerator.size, 3.0)])
        backprojected_new = np.stack([denominator, np.full(numerator.size, 2.0)])

        expected = _update(scene, backprojected, backprojected_new, 0.5)

        # uncompiled, the kernel divides scalars of numpy, which warn about
        # an infinity over an infinity where the compiled kernel does not
        with np.errstate(invalid="ignore"):
            update(scene, backprojected, backprojected_new, 0.5)

        assert np.all(np.isfinite(scene))
        assert np.any(scene == 0)
        assert np.allclose(scene, expected, rtol=1e-12)
