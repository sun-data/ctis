"""
Models of CTIS instruments used during inversions.
"""

from ._instruments import (
    AbstractInstrument,
    AbstractLinearInstrument,
    IdealInstrument,
    OptikaInstrument,
)

__all__ = [
    "AbstractInstrument",
    "AbstractLinearInstrument",
    "IdealInstrument",
    "OptikaInstrument",
]
