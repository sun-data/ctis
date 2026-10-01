import warnings
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis


def test_mean_chi_squared_zero_uncertainty():
    """
    Points with no uncertainty are excluded from the mean without a warning.
    """
    observed = na.ScalarArray(np.array([1, 2, 0]) * u.electron, axes="x")
    expected = na.ScalarArray(np.array([2, 2, 0]) * u.electron, axes="x")
    uncertainty = na.ScalarArray(np.array([1, 1, 0]) * u.electron, axes="x")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = ctis.inverters.merit.mean_chi_squared(
            observed=observed,
            expected=expected,
            uncertainty=uncertainty,
            axis="x",
        )

    assert np.allclose(result.ndarray, 0.5)
