from typing import Sequence
import numpy as np
import named_arrays as na

__all__ = [
    "mean_chi_squared",
    "correlation_residual",
]


def mean_chi_squared(
    observed: na.ScalarArray,
    expected: na.ScalarArray,
    uncertainty: na.ScalarArray,
    axis: None | str | Sequence[str] = None,
) -> na.ScalarArray:
    r"""
    Compute :math:`\langle \chi^2 \rangle = \biggl\langle \left( \frac{O - E}{\sigma} \right)^2 \biggr \rangle` ,
    where :math:`O` is the observed value,
    :math:`E` is the expected value,
    and :math:`\sigma` denotes the standard deviation of the uncertainty.

    Parameters
    ----------
    observed
        The measured values.
    expected
        The values predicted by the model.
    uncertainty
        The uncertainty of the values predicted by the model.
    axis
        The logical axis or axes over which to average the result.
    """
    chisq = np.square((observed - expected) / uncertainty)

    where = uncertainty != 0

    return np.mean(
        a=chisq,
        axis=axis,
        where=where,
    )


def correlation_residual(
    observed: na.ScalarArray,
    expected: na.ScalarArray,
    axis: None | str | Sequence[str] = None,
) -> na.ScalarArray:
    """
    Compute Pearson's correlation coefficient between the expected values
    and the residual.

    Parameters
    ----------
    observed
        The measured values.
    expected
        The values predicted by the model.
    axis
        The logical axis or axes over which to average the result.
    """

    residual = observed - expected

    r = na.stats.pearsonr(expected, residual, axis=axis)

    return r
