"""
Models of CTIS instruments used during inversions.
"""

from ._instruments import (
    AbstractInstrument,
    AbstractLinearInstrument,
    IdealInstrument,
    LinearOptikaInstrument,
)

__all__ = [
    "AbstractInstrument",
    "AbstractLinearInstrument",
    "IdealInstrument",
    "LinearOptikaInstrument",
]
