from typing import ClassVar
import abc
import dataclasses
import named_arrays as na
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


@dataclasses.dataclass
class IterativeInversionResult(
    InversionResult,
):
    """The results of an iterative inversion attempt."""

    inverter: AbstractIterativeInverter

    num_iteration: int
    """The number of iterations performed by the inverter."""

    merit: na.ScalarArray
    """The value of the merit function for each iteration."""

    merit_name: str
    """Human-readable name of the merit function."""

    @property
    def iteration(self) -> na.ScalarArray:
        """The iteration value for each iteration."""
        return na.arange(
            start=0,
            stop=self.num_iteration,
            axis=self.inverter.axis_iteration,
        )

