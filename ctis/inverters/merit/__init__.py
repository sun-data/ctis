"""Functions used to evaluate the quality of CTIS inversions."""

from ._merit import (
    mean_chi_squared,
    correlation_residual,
)

__all__ = [
    "mean_chi_squared",
    "correlation_residual",
]
