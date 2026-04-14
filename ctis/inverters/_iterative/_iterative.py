import abc
import dataclasses
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

    @property
    @abc.abstractmethod
    def num_iteration(self):
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

    num_iteration: int
    """The number of iterations performed by the inverter."""

    axis_intermediate: None | str = None
    """
    The logical axis representing potential intermediate results.
    If :obj:`None` (the default), there are no intermediate results.
    """
