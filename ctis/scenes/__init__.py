import abc
import dataclasses
import named_arrays as na
import numpy as np
import astropy.constants as const
import astropy.units as u

"""
Test patterns used to evaluate the quality of CTIS inversions.
"""

from ._gaussians import gaussians

__all__ = [
    "gaussians",
]


class Scene(na.FunctionArray):
    """
    Scene
    """

    def zeroth_moment(self, axis):
        return np.sum(self.scene.outputs, axis=axis, keepdims=False)

    def first_moment_velocity(self, wv_0):
        weight = (self.inputs.wavelength - wv_0) / wv_0 * const.c
        weight = weight.to(u.km / u.s)
        return np.sum(self.scene.outputs * weight, axis='wavelength', keepdims=False) / self.zeroth_moment(axis='wavelength')
