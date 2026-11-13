"""
Compiled kernels for the multiplicative algebraic reconstruction technique.
"""

import numpy as np
import numba

__all__ = [
    "merit",
    "update",
]


@numba.njit(cache=True, parallel=True, error_model="numpy")
def merit(
    observed: np.ndarray,
    predicted: np.ndarray,
    variance: np.ndarray,
    chi_squared: np.ndarray,
    correlation: np.ndarray,
) -> None:
    r"""
    Evaluate, for each channel, the mean :math:`\chi^2` of the predicted
    images, and Pearson's correlation coefficient between the predicted
    images and the residual.

    These are the quantities computed by
    :func:`ctis.inverters.merit.mean_chi_squared` and
    :func:`ctis.inverters.merit.correlation_residual`, in one pass over the
    pixels of each channel.
    The pixels of a channel are summed in order by a single thread, so the
    result does not depend on the number of threads.

    Parameters
    ----------
    observed
        The measured images, with one row for each channel.
    predicted
        The images predicted by the model, with one row for each channel.
    variance
        The variance of the noise in the predicted images.
        A pixel where it is zero, such as one outside the field of view, is
        left out of the mean :math:`\chi^2`.
    chi_squared
        An array with an element for each channel, which is overwritten with
        the mean :math:`\chi^2`.
    correlation
        An array with an element for each channel, which is overwritten with
        the correlation coefficient.
    """
    num_channel, num_pixel = observed.shape

    for c in numba.prange(num_channel):
        total = 0.0
        count = 0
        mean_predicted = 0.0
        mean_residual = 0.0
        for p in range(num_pixel):
            residual = observed[c, p] - predicted[c, p]
            if variance[c, p] != 0:
                total += residual * residual / variance[c, p]
                count += 1
            mean_predicted += predicted[c, p]
            mean_residual += residual
        mean_predicted /= num_pixel
        mean_residual /= num_pixel

        covariance = 0.0
        variance_predicted = 0.0
        variance_residual = 0.0
        for p in range(num_pixel):
            x = predicted[c, p] - mean_predicted
            y = observed[c, p] - predicted[c, p] - mean_residual
            covariance += x * y
            variance_predicted += x * x
            variance_residual += y * y

        # a channel with nothing to average has no merit, compiled or not
        chi_squared[c] = np.nan
        if count > 0:
            chi_squared[c] = total / count

        correlation[c] = np.nan
        denominator = np.sqrt(variance_predicted * variance_residual)
        if denominator > 0:
            correlation[c] = covariance / denominator


@numba.njit(cache=True, parallel=True, error_model="numpy")
def update(
    scene: np.ndarray,
    backprojected: np.ndarray,
    backprojected_new: np.ndarray,
    gamma: float,
) -> None:
    r"""
    Multiply every voxel of a scene by its correction, the geometric mean
    over the channels of :math:`(B / B')^\gamma`, where :math:`B` is the
    backprojection of the observed images and :math:`B'` is the
    backprojection of the predicted images.

    A negative backprojection is treated as zero.
    The correction of a channel which is not finite, such as that of a voxel
    outside the field of view of the channel, is treated as one.

    Parameters
    ----------
    scene
        The voxels of the scene, which are overwritten.
    backprojected
        The backprojection of the observed images, with one row for each
        channel.
    backprojected_new
        The backprojection of the images predicted from `scene`, with one
        row for each channel.
    gamma
        The exponent applied to the correction of each channel,
        :math:`\gamma`.
    """
    num_channel, num_voxel = backprojected.shape
    power = 1 / num_channel

    for v in numba.prange(num_voxel):
        correction = 1.0
        for c in range(num_channel):
            numerator = backprojected[c, v]
            denominator = backprojected_new[c, v]
            # a denominator which is zero, negative or not a number gives a
            # ratio which is not finite
            if denominator > 0:
                if numerator < 0:
                    numerator = 0.0
                ratio = numerator / denominator
                if np.isfinite(ratio):
                    correction *= ratio**gamma
        scene[v] *= correction**power
