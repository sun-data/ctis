import abc
import dataclass

@dataclasses.dataclass
class AbstractResults(
    abc.ABC,
):
    """
    An interface describing the results of various inversion routines.
    """
    solution: na.FunctionArray
    merit: float
    converged=None