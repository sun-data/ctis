"""
Compiled kernels for the multiplicative algebraic reconstruction technique.

Every array has the frames, the independent sets of images which are
reconstructed together, along its last axis.
Each frame is computed on its own, in the same order whatever the number of
frames, so a frame gives the same result in a batch as it does alone, bit for
bit.
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
    Evaluate, for each frame and each channel, the mean :math:`\chi^2` of the
    predicted images, and Pearson's correlation coefficient between the
    predicted images and the residual.

    These are the quantities computed by
    :func:`ctis.inverters.merit.mean_chi_squared` and
    :func:`ctis.inverters.merit.correlation_residual`, in two passes over the
    pixels of each channel.
    The pixels of a channel are summed in order by a single thread, so the
    result does not depend on the number of threads.

    Parameters
    ----------
    observed
        The measured images, an array with an axis for the channels, one for
        the pixels and one for the frames.
    predicted
        The images predicted by the model.
    variance
        The variance of the noise in the predicted images.
        A pixel where it is zero, such as one outside the field of view, is
        left out of the mean :math:`\chi^2`.
    chi_squared
        An array with an axis for the frames and one for the channels, which
        is overwritten with the mean :math:`\chi^2`.
    correlation
        An array with an axis for the frames and one for the channels, which
        is overwritten with the correlation coefficient.
    """
    num_channel, num_pixel, num_frame = observed.shape

    for c in numba.prange(num_channel):
        total = np.zeros(num_frame)
        count = np.zeros(num_frame, dtype=np.int64)
        mean_predicted = np.zeros(num_frame)
        mean_residual = np.zeros(num_frame)
        for p in range(num_pixel):
            for f in range(num_frame):
                expected = np.float64(predicted[c, p, f])
                residual = np.float64(observed[c, p, f]) - expected
                noise = np.float64(variance[c, p, f])
                if noise != 0:
                    total[f] += residual * residual / noise
                    count[f] += 1
                mean_predicted[f] += expected
                mean_residual[f] += residual
        for f in range(num_frame):
            mean_predicted[f] /= num_pixel
            mean_residual[f] /= num_pixel

        covariance = np.zeros(num_frame)
        variance_predicted = np.zeros(num_frame)
        variance_residual = np.zeros(num_frame)
        for p in range(num_pixel):
            for f in range(num_frame):
                expected = np.float64(predicted[c, p, f])
                residual = np.float64(observed[c, p, f]) - expected
                x = expected - mean_predicted[f]
                y = residual - mean_residual[f]
                covariance[f] += x * y
                variance_predicted[f] += x * x
                variance_residual[f] += y * y

        for f in range(num_frame):
            # a channel with nothing to average has no merit, compiled or not
            chi_squared[f, c] = np.nan
            if count[f] > 0:
                chi_squared[f, c] = total[f] / count[f]

            correlation[f, c] = np.nan
            denominator = np.sqrt(variance_predicted[f] * variance_residual[f])
            if denominator > 0:
                correlation[f, c] = covariance[f] / denominator


@numba.njit(cache=True, parallel=True, error_model="numpy")
def update(
    scene: np.ndarray,
    backprojected: np.ndarray,
    backprojected_new: np.ndarray,
    gamma: float,
) -> None:
    r"""
    Multiply every voxel of the scene of each frame by its correction, the
    geometric mean over the channels of :math:`(B / B')^\gamma`,
    where :math:`B` is the backprojection of the observed images and
    :math:`B'` is the backprojection of the predicted images.

    A negative backprojection is treated as zero.
    The correction of a channel which is not finite, such as that of a voxel
    outside the field of view of the channel, is treated as one.

    Parameters
    ----------
    scene
        The scenes, an array with an axis for the voxels and one for the
        frames, which is overwritten.
    backprojected
        The backprojection of the observed images, an array with an axis for
        the channels, one for the voxels and one for the frames.
    backprojected_new
        The backprojection of the images predicted from `scene`.
    gamma
        The exponent applied to the correction of each channel,
        :math:`\gamma`.
    """
    num_channel, num_voxel, num_frame = backprojected.shape
    power = 1 / num_channel

    for v in numba.prange(num_voxel):
        for f in range(num_frame):
            correction = 1.0
            for c in range(num_channel):
                numerator = np.float64(backprojected[c, v, f])
                denominator = np.float64(backprojected_new[c, v, f])
                # a denominator which is zero, negative or not a number gives
                # a ratio which is not finite
                if denominator > 0:
                    if numerator < 0:
                        numerator = 0.0
                    ratio = numerator / denominator
                    if np.isfinite(ratio):
                        correction *= ratio**gamma
            scene[v, f] *= correction**power
