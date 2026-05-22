from typing import ClassVar
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
from .. import AbstractInverter, AbstractInversionResult

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

    num_iteration: int = dataclasses.field(default=100, kw_only=True)
    """
    The maximum number of iterations to perform.

    If convergence is not reached before this number is exceeded,
    a warning is raised and an unsuccessful result is returned. 
    """

    intermediate: bool = dataclasses.field(default=False, kw_only=True)
    """
    Whether to save intermediate solutions.

    This is set to :obj:`False` during normal operation, but can be useful for
    debugging or demonstration purposes.
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
    AbstractInversionResult,
):
    """The results of an iterative inversion attempt."""

    solutions: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]
    """
    Intermediate solutions from each iteration.
    
    If :attr:`AbstractIterativeInverter.intermediate` is set to :obj:`True`,
    this has up to :attr:`~AbstractIterativeInverter.num_iteration` elements
    along the :attr:`~AbstractIterativeInverter.axis_iteration` logical axis.
    Otherwise this has only one element along the
    :attr:`~AbstractIterativeInverter.axis_iteration` axis.
    """

    success: bool = dataclasses.MISSING
    """A boolean flag indicating whether the inversion was successful."""

    images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray] = dataclasses.MISSING
    """The observed images on which the inversion was performed."""

    inverter: "ctis.inverters.AbstractInverter" = dataclasses.MISSING
    """The inversion algorithm instance that produced these results."""

    message: str = dataclasses.MISSING
    """Any message from the inversion routine concerning the results."""

    num_iteration: int = dataclasses.MISSING
    """The number of iterations performed by the inverter."""

    mean_chi_squared: na.ScalarArray = dataclasses.MISSING
    """The mean chi squared statistic for each iteration."""

    correlation_residual: na.ScalarArray = dataclasses.MISSING
    """
    The correlation between the predicted images and the residuals 
    for each iteration.
    """

    @property
    def solution(
        self,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
        axis_iteration = self.inverter.axis_iteration
        return self.solutions[{axis_iteration: ~0}]

    @property
    def iteration(self) -> na.ScalarArray:
        """The iteration value for each iteration."""
        return na.arange(
            start=0,
            stop=self.num_iteration,
            axis=self.inverter.axis_iteration,
        )
