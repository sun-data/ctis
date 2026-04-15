import dataclasses
import named_arrays as na
import ctis

__all__ = [
    "InversionResult",
]

@dataclasses.dataclass
class InversionResult:
    """
    The results of an inversion attempt.
    """

    solution: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]
    """The reconstructed scene found by the inversion."""

    success: bool
    """A boolean flag indicating whether the inversion was successful."""

    images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]
    """The observed images on which the inversion was performed."""

    inverter: 'ctis.inverters.AbstractInverter'
    """The inversion algorithm that produced these results."""

    message: str
    """Any message from the inversion routine concerning the results."""
