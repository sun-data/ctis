"""
A :mod:`torch` backend for the CTIS forward model.

The forward model of a :class:`~ctis.instruments.AbstractLinearInstrument` is a
sparse matrix multiplication, which :mod:`regridding` stores as a collection of
``(indices_input, indices_output, values)`` triplets.
:class:`Regridder` assembles those triplets into a single sparse CSR matrix
which can be applied on a GPU and differentiated by :mod:`torch`.

Notes
-----
The compressed sparse row (CSR) format is used instead of the more obvious
scatter-add (:meth:`torch.Tensor.index_add_`) since it is both faster and,
unlike scatter-add, bitwise deterministic on CUDA devices.
Determinism matters because the inversion routines built on this module solve
linear systems using conjugate gradient methods, which assume the operator
does not change between applications.

Since the adjoint of this operator is computed by :mod:`torch` automatic
differentiation, it is the exact transpose of the forward model.
This is *not* the same as
:meth:`~ctis.instruments.AbstractInstrument.backproject`, which applies an
additional normalization to conserve flux.
"""

from typing import TYPE_CHECKING
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "Regridder",
]

if TYPE_CHECKING:  # pragma: nocover
    import torch


def _torch():
    """
    Import :mod:`torch` lazily so that it remains an optional dependency.
    """
    try:
        import torch
    except ImportError as e:  # pragma: nocover
        raise ImportError(
            "PyTorch is required to use `ctis._torch`. "
            "Install it using `pip install ctis[torch]`."
        ) from e
    return torch


@dataclasses.dataclass(eq=False)
class Regridder:
    """
    A sparse linear operator which resamples values from an input grid onto an
    output grid using :mod:`torch`.

    This is a :mod:`torch` analogue of
    :func:`regridding.regrid_from_weights`, except that the sparse matrix is
    assembled once and applied many times, and the result is differentiable.

    Examples
    --------

    Project a uniform scene onto the sensors of an ideal CTIS instrument.

    .. jupyter-execute::

        import numpy as np
        import astropy.units as u
        import named_arrays as na
        import ctis

        # Define the grid of velocities and positions on the skyplane
        coordinates_scene = na.DopplerPositionalVectorArray.from_velocity(
            velocity=na.linspace(-300, 300, axis="wavelength", num=6) * u.km / u.s,
            wavelength_rest=630 * u.AA,
            position=na.Cartesian2dVectorLinearSpace(
                start=-5 * u.arcsec,
                stop=5 * u.arcsec,
                axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
                num=17,
            ),
        )

        # Define the grid of positions on the sensor
        coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
            velocity=coordinates_scene.velocity,
            wavelength_rest=630 * u.AA,
            position=na.Cartesian2dVectorArray(
                x=na.arange(0, 33, axis="sensor_x") * u.pix,
                y=na.arange(0, 33, axis="sensor_y") * u.pix,
            ),
        )

        # Define an idealized CTIS instrument with two channels
        angle = na.linspace(0, 180, axis="channel", num=2, endpoint=False) * u.deg
        instrument = ctis.instruments.IdealInstrument(
            area_effective=1 * u.cm**2,
            timedelta_exposure=10 * u.s,
            plate_scale=0.5 * u.arcsec / u.pix,
            dispersion=0.01 * u.AA / u.pix,
            angle=angle,
            wavelength_ref=630 * u.AA,
            position_ref=16 * u.pix,
            coordinates_scene=coordinates_scene,
            coordinates_sensor=coordinates_sensor,
            channel=angle,
            axis_channel="channel",
            axis_wavelength="wavelength",
            axis_scene_xy=("scene_x", "scene_y"),
            axis_sensor_xy=("sensor_x", "sensor_y"),
        )

        # Assemble the sparse forward operator
        regridder = ctis.Regridder.from_weights(
            weights=instrument.weights,
            axis_input=instrument.axis_scene_xy,
            axis_output=instrument.axis_sensor_xy,
        )

        # Project a uniform scene onto the sensors.
        # The operator is placed on a CUDA device if one is available, so the
        # values must be created on `regridder.device` to match.
        import torch
        scene = torch.ones(
            regridder.shape_values_input,
            device=regridder.device,
        )
        image = regridder(scene)

        image.shape
    """

    matrix: "torch.Tensor" = dataclasses.MISSING
    """The sparse CSR matrix representing this operator."""

    axis_block: tuple[str, ...] = dataclasses.MISSING
    """
    The logical axes along which this operator is block diagonal.

    These are the axes of the array of weights computed by
    :func:`regridding.weights`, usually the wavelength and channel axes.
    """

    axis_input: tuple[str, ...] = dataclasses.MISSING
    """The logical axes of the input grid which are resampled."""

    axis_output: tuple[str, ...] = dataclasses.MISSING
    """The logical axes of the output grid which are resampled."""

    shape_input: dict[str, int] = dataclasses.MISSING
    """The shape of the input grid."""

    shape_output: dict[str, int] = dataclasses.MISSING
    """The shape of the output grid."""

    unit: None | u.UnitBase = None
    """
    The unit of the weights, if they carry one.

    The values returned by :meth:`__call__` are plain :mod:`torch` tensors,
    so the caller is responsible for reapplying this unit.
    """

    @classmethod
    def from_weights(
        cls,
        weights: tuple["na.AbstractScalar", dict[str, int], dict[str, int]],
        axis_input: str | tuple[str, ...],
        axis_output: str | tuple[str, ...],
        device: None | str = None,
        dtype: "None | torch.dtype" = None,
    ) -> "Regridder":
        """
        Assemble a sparse operator from weights computed by
        :func:`regridding.weights`.

        Parameters
        ----------
        weights
            The weights computed by :func:`regridding.weights`, usually the
            :attr:`~ctis.instruments.AbstractLinearInstrument.weights` attribute
            of an instrument.
        axis_input
            The logical axes of the input grid which are resampled, usually the
            :attr:`~ctis.instruments.AbstractInstrument.axis_scene_xy` attribute
            of an instrument.
        axis_output
            The logical axes of the output grid which are resampled, usually the
            :attr:`~ctis.instruments.AbstractInstrument.axis_sensor_xy` attribute
            of an instrument.
        device
            The :mod:`torch` device on which to place the matrix.
            If :obj:`None`, a CUDA device is used if one is available.
        dtype
            The floating-point type of the matrix.
            If :obj:`None`, :obj:`torch.float32` is used.
            Double precision is very slow on consumer GPUs and is usually
            not needed since the matrix elements are geometric areas.
        """
        torch = _torch()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype is None:
            dtype = torch.float32

        array, shape_input, shape_output = weights

        if isinstance(axis_input, str):  # pragma: nocover
            axis_input = (axis_input,)
        if isinstance(axis_output, str):  # pragma: nocover
            axis_output = (axis_output,)

        axis_input = tuple(axis_input)
        axis_output = tuple(axis_output)
        axis_block = tuple(array.axes)

        num_input = int(np.prod([shape_input[ax] for ax in axis_input]))
        num_output = int(np.prod([shape_output[ax] for ax in axis_output]))

        flat = array.ndarray.reshape(-1)
        num_block = flat.size

        rows = []
        columns = []
        values = []
        unit = None

        for d in range(num_block):
            indices_input, indices_output, values_d = flat[d]
            unit_d = getattr(values_d, "unit", None)
            if unit_d is not None:
                unit = unit_d
                values_d = values_d.value
            rows.append(np.asarray(indices_output) + d * num_output)
            columns.append(np.asarray(indices_input) + d * num_input)
            values.append(np.asarray(values_d))

        rows = np.concatenate(rows)
        columns = np.concatenate(columns)
        values = np.concatenate(values)

        size = (num_block * num_output, num_block * num_input)

        if max(size) > np.iinfo(np.int32).max:  # pragma: nocover
            raise ValueError(
                f"the operator shape {size} is too large for 32-bit indices."
            )

        # sort by row so the matrix can be stored in CSR format, which is both
        # faster and deterministic on CUDA devices.
        order = np.argsort(rows, kind="stable")
        columns = columns[order].astype(np.int32)
        values = values[order]

        crow = np.zeros(size[0] + 1, dtype=np.int32)
        crow[1:] = np.cumsum(np.bincount(rows, minlength=size[0]))

        matrix = torch.sparse_csr_tensor(
            crow_indices=torch.as_tensor(crow, device=device),
            col_indices=torch.as_tensor(columns, device=device),
            values=torch.as_tensor(values, device=device).to(dtype),
            size=size,
        )

        return cls(
            matrix=matrix,
            axis_block=axis_block,
            axis_input=axis_input,
            axis_output=axis_output,
            shape_input=shape_input,
            shape_output=shape_output,
            unit=unit,
        )

    @property
    def axes_values_input(self) -> tuple[str, ...]:
        """The logical axes expected by :meth:`__call__`, in order."""
        return self.axis_block + self.axis_input

    @property
    def axes_values_output(self) -> tuple[str, ...]:
        """The logical axes returned by :meth:`__call__`, in order."""
        return self.axis_block + self.axis_output

    @property
    def shape_values_input(self) -> tuple[int, ...]:
        """The shape of the array expected by :meth:`__call__`."""
        return tuple(self.shape_input[ax] for ax in self.axes_values_input)

    @property
    def shape_values_output(self) -> tuple[int, ...]:
        """The shape of the array returned by :meth:`__call__`."""
        return tuple(self.shape_output[ax] for ax in self.axes_values_output)

    @property
    def device(self) -> "torch.device":
        """The device on which this operator is stored."""
        return self.matrix.device

    @property
    def dtype(self) -> "torch.dtype":
        """The floating-point type of this operator."""
        return self.matrix.dtype

    def __call__(self, values: "torch.Tensor") -> "torch.Tensor":
        """
        Resample an array of values from the input grid onto the output grid.

        Parameters
        ----------
        values
            The values to resample.
            The trailing axes must match :attr:`shape_values_input`, and any
            leading axes are treated as batch axes.
        """
        shape_input = self.shape_values_input
        shape_output = self.shape_values_output

        ndim = len(shape_input)

        if tuple(values.shape[values.ndim - ndim :]) != shape_input:
            raise ValueError(
                f"the trailing axes of {tuple(values.shape)=} should match "
                f"{shape_input}."
            )

        shape_batch = tuple(values.shape[: values.ndim - ndim])

        num_input = self.matrix.shape[1]

        result = values.reshape(*shape_batch, num_input)

        if shape_batch:
            result = result.reshape(-1, num_input).transpose(0, 1)
            result = (self.matrix @ result).transpose(0, 1)
        else:
            result = self.matrix @ result

        return result.reshape(*shape_batch, *shape_output)
