import abc
import pytest
import numpy as np
import astropy.units as u
import ctis

torch = pytest.importorskip("torch")


velocity = torch.linspace(-400, 400, 33, dtype=torch.float64)


class AbstractTestAbstractSpectralModel(
    abc.ABC,
):

    def test_parameters(self, a: ctis.inverters.AbstractSpectralModel):
        result = a.parameters
        assert isinstance(result, tuple)
        assert len(result) > 0
        for name in result:
            assert isinstance(name, str)

    def test_num_parameters(self, a: ctis.inverters.AbstractSpectralModel):
        result = a.num_parameters
        assert isinstance(result, int)
        assert result == len(a.parameters)

    def test_physical(self, a: ctis.inverters.AbstractSpectralModel):
        parameters = torch.zeros((a.num_parameters, 3, 4), dtype=torch.float64)
        result = a.physical(parameters)

        assert isinstance(result, dict)
        for name in a.parameters:
            assert name in result
            assert result[name].shape == (3, 4)

    def test_guess(self, a: ctis.inverters.AbstractSpectralModel):
        """`guess` must be the inverse of `physical`."""
        parameters = torch.zeros((a.num_parameters, 3, 4), dtype=torch.float64)
        physical = a.physical(parameters)

        result = a.guess(**{k: physical[k] for k in a.parameters})

        assert result.shape == parameters.shape
        assert torch.allclose(result, parameters, atol=1e-6)

    def test__call__(self, a: ctis.inverters.AbstractSpectralModel):
        parameters = torch.zeros((a.num_parameters, 3, 4), dtype=torch.float64)
        result = a(parameters, velocity)

        assert result.shape == (velocity.numel() - 1, 3, 4)
        assert torch.all(result >= 0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        ctis.inverters.GaussianModel(),
        ctis.inverters.GaussianModel(
            width_thermal=11 * u.km / u.s,
            width_instrument=8 * u.km / u.s,
            velocity_max=250 * u.km / u.s,
        ),
    ],
)
class TestGaussianModel(
    AbstractTestAbstractSpectralModel,
):
    def test_parameters_names(self, a: ctis.inverters.GaussianModel):
        assert a.parameters == ("intensity", "velocity", "width_nonthermal")

    @pytest.mark.parametrize("intensity", [1.0, 1234.0])
    @pytest.mark.parametrize("shift", [-100.0, 0.0, 50.0])
    @pytest.mark.parametrize("width_nonthermal", [20.0, 60.0])
    def test__call__normalization(
        self,
        a: ctis.inverters.GaussianModel,
        intensity: float,
        shift: float,
        width_nonthermal: float,
    ):
        """
        Integrating the profile over a velocity range which covers the whole
        line must return the integrated radiance.
        """
        parameters = a.guess(
            intensity=torch.tensor(intensity, dtype=torch.float64),
            velocity=torch.tensor(shift, dtype=torch.float64),
            width_nonthermal=torch.tensor(width_nonthermal, dtype=torch.float64),
        )

        result = a(parameters, velocity)

        width_bin = torch.diff(velocity)
        integral = (result * width_bin).sum()

        assert np.isclose(integral.item(), intensity, rtol=1e-6)

    def test__call__bin_integrated(self, a: ctis.inverters.GaussianModel):
        """
        The profile must be the mean of the underlying Gaussian across each
        bin, not the Gaussian sampled at the bin center.
        """
        parameters = a.guess(
            intensity=torch.tensor(1.0, dtype=torch.float64),
            velocity=torch.tensor(30.0, dtype=torch.float64),
            width_nonthermal=torch.tensor(25.0, dtype=torch.float64),
        )

        result = a(parameters, velocity).numpy()

        physical = a.physical(parameters)
        shift = physical["velocity"].item()
        width = physical["width"].item()

        # numerically integrate the Gaussian across each bin
        edges = velocity.numpy()
        expected = np.empty(edges.size - 1)
        for i in range(expected.size):
            v = np.linspace(edges[i], edges[i + 1], 201)
            g = np.exp(-np.square((v - shift) / width) / 2)
            g = g / (width * np.sqrt(2 * np.pi))
            expected[i] = np.trapezoid(g, v) / (edges[i + 1] - edges[i])

        assert np.allclose(result, expected, atol=1e-9)

    def test_width(self, a: ctis.inverters.GaussianModel):
        """The total width must include the fixed thermal and instrument widths."""
        parameters = torch.zeros((a.num_parameters, 2), dtype=torch.float64)
        physical = a.physical(parameters)

        width_thermal = a.width_thermal.to_value(u.km / u.s)
        width_instrument = a.width_instrument.to_value(u.km / u.s)

        expected = np.sqrt(
            width_thermal**2
            + width_instrument**2
            + physical["width_nonthermal"].numpy() ** 2
        )

        assert np.allclose(physical["width"].numpy(), expected)
        assert np.all(physical["width"].numpy() >= physical["width_nonthermal"].numpy())
