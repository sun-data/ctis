from typing import TYPE_CHECKING
import abc
import dataclasses
import math
import astropy.units as u
from ..._torch import _torch

__all__ = [
    "AbstractSpectralModel",
    "GaussianModel",
]

if TYPE_CHECKING:  # pragma: nocover
    import torch


@dataclasses.dataclass
class AbstractSpectralModel(
    abc.ABC,
):
    """
    An interface describing a parameterized model of the spectral line profile
    observed in each spatial pixel of a scene.

    Models are expressed in Doppler velocity space since the rest wavelength
    of the observed line is usually known from atomic physics.

    The optimizer which fits these models operates on `unconstrained`
    parameters, which are mapped onto `physical` parameters by :meth:`physical`.
    This allows bounds such as the positivity of the line intensity to be
    expressed exactly instead of as a penalty.

    Implementations should choose link functions which make every
    unconstrained parameter of order unity, since gradient-based optimizers
    take steps of a fixed size in the unconstrained space.
    Positive quantities which span orders of magnitude, such as the line
    radiance, should therefore be parameterized logarithmically rather than
    with a softplus.
    """

    @property
    @abc.abstractmethod
    def parameters(self) -> tuple[str, ...]:
        """The name of each physical parameter of this model."""

    @property
    def num_parameters(self) -> int:
        """The number of free parameters of this model, per spatial pixel."""
        return len(self.parameters)

    @abc.abstractmethod
    def unit(self, intensity: u.UnitBase) -> dict[str, u.UnitBase]:
        """
        The unit of each quantity returned by :meth:`physical`.

        Parameters
        ----------
        intensity
            The unit of the line radiance integrated over the spectral line.
            This is determined by the instrument rather than by the model, so
            it is supplied by the caller.
        """

    @abc.abstractmethod
    def physical(
        self,
        parameters: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        """
        Map unconstrained parameters onto physical parameters.

        Parameters
        ----------
        parameters
            The unconstrained parameters seen by the optimizer.
            The leading axis has :attr:`num_parameters` elements and the
            remaining axes are the spatial axes of the scene.
        """

    @abc.abstractmethod
    def guess(
        self,
        **kwargs: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Map physical parameters onto unconstrained parameters.

        This is the inverse of :meth:`physical`, and is used to convert an
        initial guess expressed in physical units into a starting point for
        the optimizer.

        Parameters
        ----------
        kwargs
            The physical parameters, named according to :attr:`parameters`.
        """

    @abc.abstractmethod
    def __call__(
        self,
        parameters: "torch.Tensor",
        velocity: "torch.Tensor",
    ) -> "torch.Tensor":
        r"""
        Evaluate the mean spectral radiance within each velocity bin.

        The profile is integrated `analytically` across each bin rather than
        sampled at the bin center.
        This is essential, not cosmetic: the reconstruction grid is usually
        comparable to or coarser than the width of the spectral line, so
        sampling at bin centers would make a sub-bin Doppler shift almost
        unobservable.

        Parameters
        ----------
        parameters
            The unconstrained parameters seen by the optimizer,
            with shape ``(num_parameters, ...)``.
        velocity
            The edges of each velocity bin, in units of
            :math:`\text{km} \, \text{s}^{-1}`.
        """


@dataclasses.dataclass
class GaussianModel(
    AbstractSpectralModel,
):
    r"""
    A single Gaussian spectral line profile.

    The radiance within the velocity bin spanning :math:`[v_0, v_1]` is

    .. math::

        \frac{I}{2 (v_1 - v_0)} \left[
            \text{erf} \left( \frac{v_1 - v}{\sqrt{2} \sigma} \right)
            - \text{erf} \left( \frac{v_0 - v}{\sqrt{2} \sigma} \right)
        \right],

    The unconstrained parameters seen by the optimizer are mapped onto the
    physical parameters using

    .. math::

        I = e^{\theta_0}, \quad
        v = v_\text{max} \tanh \theta_1, \quad
        \sigma_\text{nonthermal} = e^{\theta_2},

    where :math:`I` is the radiance integrated over the line,
    :math:`v` is the bulk Doppler velocity,
    and the total width is

    .. math::

        \sigma^2 = \sigma_\text{thermal}^2
                 + \sigma_\text{instrument}^2
                 + \sigma_\text{nonthermal}^2.

    Since :attr:`width_thermal` is fixed by the mass of the emitting ion and
    the formation temperature of the line, and :attr:`width_instrument` is
    fixed by the instrument, the only free width parameter is the nonthermal
    width.
    Fitting the nonthermal width directly means that the physically
    interesting quantity is the one which receives an uncertainty, instead of
    being recovered afterwards from a difference of comparable squares.
    It also makes the lower bound on the observed width exact rather than a
    penalty.
    """

    width_thermal: u.Quantity = 0 * u.km / u.s
    r"""
    The thermal Doppler width, :math:`\sigma_\text{thermal}`, of the observed
    spectral line.

    This is :math:`\sqrt{k_\text{B} T / m}`, where :math:`T` is the formation
    temperature of the line and :math:`m` is the mass of the emitting ion.
    """

    width_instrument: u.Quantity = 0 * u.km / u.s
    r"""
    The width, :math:`\sigma_\text{instrument}`, of the instrument's spectral
    response function.
    """

    velocity_max: u.Quantity = 300 * u.km / u.s
    r"""
    The maximum magnitude of the bulk Doppler velocity.

    The velocity is parameterized as
    :math:`v = v_\text{max} \tanh(\theta)`, which bounds the fit to the
    passband and prevents the optimizer from wandering into aliased solutions.
    """

    @property
    def parameters(self) -> tuple[str, ...]:
        return ("intensity", "velocity", "width_nonthermal")

    @property
    def _velocity_max(self) -> float:
        return self.velocity_max.to_value(u.km / u.s)

    @property
    def _width_fixed_squared(self) -> float:
        """The known contributions to the variance of the line profile."""
        width_thermal = self.width_thermal.to_value(u.km / u.s)
        width_instrument = self.width_instrument.to_value(u.km / u.s)
        return width_thermal**2 + width_instrument**2

    def unit(self, intensity: u.UnitBase) -> dict[str, u.UnitBase]:
        velocity = u.km / u.s
        return dict(
            intensity=intensity,
            velocity=velocity,
            width_nonthermal=velocity,
            width=velocity,
        )

    def physical(
        self,
        parameters: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        torch = _torch()

        intensity = torch.exp(parameters[0])
        velocity = self._velocity_max * torch.tanh(parameters[1])
        width_nonthermal = torch.exp(parameters[2])

        width = torch.sqrt(self._width_fixed_squared + width_nonthermal**2)

        return dict(
            intensity=intensity,
            velocity=velocity,
            width_nonthermal=width_nonthermal,
            width=width,
        )

    def guess(
        self,
        intensity: "torch.Tensor" = None,
        velocity: "torch.Tensor" = None,
        width_nonthermal: "torch.Tensor" = None,
    ) -> "torch.Tensor":
        torch = _torch()

        velocity_max = self._velocity_max

        velocity = torch.clamp(
            velocity / velocity_max,
            min=-1 + 1e-6,
            max=+1 - 1e-6,
        )

        tiny = torch.finfo(intensity.dtype).tiny

        return torch.stack(
            [
                torch.log(torch.clamp(intensity, min=tiny)),
                torch.atanh(velocity),
                torch.log(torch.clamp(width_nonthermal, min=tiny)),
            ]
        )

    def __call__(
        self,
        parameters: "torch.Tensor",
        velocity: "torch.Tensor",
    ) -> "torch.Tensor":
        torch = _torch()

        p = self.physical(parameters)

        intensity = p["intensity"]
        shift = p["velocity"]
        width = p["width"]

        shape = (-1,) + (1,) * intensity.ndim

        lower = velocity[:-1].reshape(shape)
        upper = velocity[+1:].reshape(shape)

        scale = math.sqrt(2) * width

        result = torch.erf((upper - shift) / scale)
        result = result - torch.erf((lower - shift) / scale)

        return intensity * result / (2 * (upper - lower))
