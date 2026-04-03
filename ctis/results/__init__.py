import abc
import dataclass

@dataclasses.dataclass
class AbstractResults(
    abc.ABC,
):
    """
    An interface describing the results of various inversion routines.
    """

    @property
    @abc.abstractmethod
    def solution(self) -> na.FunctionArray:
        """
        Inverted scene
        """

    @property
    @abc.abstractmethod
    def convergence_message(self) -> str:
        """
        Exit message of inverted communicating success/failure
        """

    @property
    @abc.abstractmethod
    def merit(self) -> float:
        """
        Value of function evaluated to determine convergence
        """

