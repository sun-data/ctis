from typing import ClassVar
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from .. import AbstractInverter, InversionResult

__all__ = [
    "AbstractIterativeInverter",
    "IterativeInversionResult",
]


@dataclasses.dataclass
class AbstractIterativeInverter(
    AbstractInverter,
):
    """
    An abstract inversion algorithm which reconstructs an observed scene
    using iterative methods.

    These methods will apply some operation repeatedly until a specified
    convergence criteria is met.
    """

    axis_iteration: ClassVar[str] = "iteration"
    """The logical axis associated with changing iteration index."""

    @property
    @abc.abstractmethod
    def num_iteration(self) -> int:
        """
        The maximum number of iterations to perform.

        If convergence is not reached before this number is exceeded,
        a warning is raised and an unsuccessful result is returned.
        """

    def mean_chi_squared(
        self,
        images_observed: na.ScalarArray,
        images_predicted: na.ScalarArray,
    ) -> na.ScalarArray:
        r"""
        Evaluate :math:`\langle \chi^2 \rangle` for each observed/predicted
        image pair.

        Parameters
        ----------
        images_observed
            The actual measured images.
        images_predicted
            The images predicted by the inversion.
        """

        uncertainty = self.instrument.uncertainty(images_predicted)

        uncertainty = np.maximum(uncertainty, 1 * u.photon)

        return ctis.inverters.merit.mean_chi_squared(
            observed=images_observed,
            expected=images_predicted,
            uncertainty=uncertainty,
            axis=self.instrument.axis_sensor_xy,
        )

    def correlation_residual(
        self,
        images_observed: na.ScalarArray,
        images_predicted: na.ScalarArray,
    ) -> na.ScalarArray:
        """
        Evaluate the correlation between the predicted images and the residual.

        Parameters
        ----------
        images_observed
            The actual measured images.
        images_predicted
            The images predicted by the inversion.
        """
        return ctis.inverters.merit.correlation_residual(
            observed=images_observed,
            expected=images_predicted,
            axis=self.instrument.axis_sensor_xy,
        )


@dataclasses.dataclass
class IterativeInversionResult(
    InversionResult,
):
    """The results of an iterative inversion attempt."""

    inverter: AbstractIterativeInverter

    num_iteration: int
    """The number of iterations performed by the inverter."""

    mean_chi_squared: na.ScalarArray
    """The mean chi squared statistic for each iteration."""

    correlation_residual: na.ScalarArray
    """
    The correlation between the predicted images and the residuals 
    for each iteration.
    """

    @property
    def iteration(self) -> na.ScalarArray:
        """The iteration value for each iteration."""
        return na.arange(
            start=0,
            stop=self.num_iteration,
            axis=self.inverter.axis_iteration,
        )
