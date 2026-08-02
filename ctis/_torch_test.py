import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis

torch = pytest.importorskip("torch")


wavelength_rest = 630 * u.AA

velocity = na.linspace(-300, 300, axis="wavelength", num=6) * u.km / u.s

coordinates_scene = na.DopplerPositionalVectorArray.from_velocity(
    velocity=velocity,
    wavelength_rest=wavelength_rest,
    position=na.Cartesian2dVectorLinearSpace(
        start=-4 * u.arcsec,
        stop=4 * u.arcsec,
        axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
        num=17,
    ),
)

coordinates_sensor = na.DopplerPositionalVectorArray.from_velocity(
    velocity=velocity,
    wavelength_rest=wavelength_rest,
    position=na.Cartesian2dVectorArray(
        x=na.arange(0, 33, axis="sensor_x") * u.pix,
        y=na.arange(0, 33, axis="sensor_y") * u.pix,
    ),
)

angle = na.linspace(0, 180, num=2, axis="channel", endpoint=False) * u.deg

instrument = ctis.instruments.IdealInstrument(
    area_effective=1 * u.cm**2,
    timedelta_exposure=10 * u.s,
    plate_scale=0.5 * u.arcsec / u.pix,
    dispersion=0.02 * u.AA / u.pix,
    angle=angle,
    wavelength_ref=wavelength_rest,
    position_ref=16 * u.pix,
    coordinates_scene=coordinates_scene,
    coordinates_sensor=coordinates_sensor,
    channel=angle,
    axis_channel="channel",
    axis_wavelength="wavelength",
    axis_scene_xy=("scene_x", "scene_y"),
    axis_sensor_xy=("sensor_x", "sensor_y"),
)


def _regridder(device: str = "cpu", dtype=None) -> ctis.Regridder:
    return ctis.Regridder.from_weights(
        weights=instrument.weights,
        axis_input=instrument.axis_scene_xy,
        axis_output=instrument.axis_sensor_xy,
        device=device,
        dtype=dtype,
    )


def _values(regridder: ctis.Regridder, seed: int = 42) -> na.ScalarArray:
    rng = np.random.default_rng(seed)
    return na.ScalarArray(
        ndarray=rng.random(regridder.shape_values_input),
        axes=regridder.axes_values_input,
    )


devices = ["cpu"]
if torch.cuda.is_available():  # pragma: nocover
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
class TestRegridder:
    def test_shape(self, device: str):
        a = _regridder(device)

        assert a.axes_values_input == ("wavelength", "channel") + tuple(
            instrument.axis_scene_xy
        )
        assert a.axes_values_output == ("wavelength", "channel") + tuple(
            instrument.axis_sensor_xy
        )

        assert a.shape_values_input == (5, 2, 16, 16)
        assert a.shape_values_output == (5, 2, 32, 32)

        assert a.matrix.shape == (5 * 2 * 32 * 32, 5 * 2 * 16 * 16)
        assert str(a.device).startswith(device)
        assert a.dtype == torch.float32

    def test__call__(self, device: str):
        """The torch operator must agree with the numba implementation."""
        a = _regridder(device)

        values = _values(a)

        expected = na.regridding.regrid_from_weights(
            *instrument.weights,
            values_input=values,
        )
        expected = expected.ndarray_aligned(a.axes_values_output)

        result = a(torch.as_tensor(values.ndarray, dtype=a.dtype, device=device))
        result = result.detach().cpu().numpy()

        assert result.shape == expected.shape
        assert np.allclose(result, expected, rtol=1e-5)

    def test__call__invalid(self, device: str):
        a = _regridder(device)
        with pytest.raises(ValueError):
            a(torch.zeros((3, 3), device=device))

    def test__call__deterministic(self, device: str):
        """
        The result must be bitwise reproducible so that conjugate gradient
        methods see a consistent operator.
        """
        a = _regridder(device)
        x = torch.as_tensor(
            _values(a).ndarray,
            dtype=a.dtype,
            device=device,
        )

        expected = a(x)
        for _ in range(8):
            assert torch.equal(a(x), expected)

    def test__call__batch(self, device: str):
        """Leading axes are treated as independent batch elements."""
        a = _regridder(device)

        x = torch.as_tensor(
            np.stack([_values(a, seed=s).ndarray for s in range(3)]),
            dtype=a.dtype,
            device=device,
        )

        result = a(x)

        assert result.shape == (3,) + a.shape_values_output

        for i in range(3):
            assert torch.allclose(result[i], a(x[i]))

    def test_adjoint(self, device: str):
        r"""
        Automatic differentiation must give the exact transpose,
        :math:`\langle u, A x \rangle = \langle A^T u, x \rangle`.
        """
        a = _regridder(device)

        rng = np.random.default_rng(0)

        x = torch.as_tensor(
            _values(a).ndarray,
            dtype=a.dtype,
            device=device,
        ).requires_grad_(True)

        u_ = torch.as_tensor(
            rng.random(a.shape_values_output),
            dtype=a.dtype,
            device=device,
        )

        y = a(x)

        lhs = (u_ * y).sum()
        (transpose,) = torch.autograd.grad(y, x, grad_outputs=u_)
        rhs = (transpose * x).sum()

        assert torch.allclose(lhs, rhs, rtol=1e-4)

    def test_unit(self, device: str):
        a = _regridder(device)
        assert a.unit is None or isinstance(a.unit, u.UnitBase)


def test_gradcheck():
    """Verify the autograd path against finite differences in double precision."""
    a = _regridder("cpu", dtype=torch.float64)

    x = torch.as_tensor(
        _values(a).ndarray,
        dtype=torch.float64,
    ).requires_grad_(True)

    assert torch.autograd.gradcheck(a, (x,), eps=1e-6, atol=1e-6)
