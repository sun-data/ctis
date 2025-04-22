import pytest
import astropy.units as u
import named_arrays as na
import ctis


@pytest.mark.parametrize(
    argnames="inputs",
    argvalues=[
        na.SpectralPositionalVectorArray(
            wavelength=na.linspace(-500, 500, axis="wavelength", num=101) * u.km / u.s,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("x", "y"),
                num=41,
            ),
        )
    ],
)
@pytest.mark.parametrize(
    argnames="width",
    argvalues=[
        na.SpectralPositionalVectorArray(
            wavelength=30 * u.km / u.s,
            position=1 * u.arcsec,
        )
    ],
)
def test_random_gaussians(
    inputs: na.AbstractSpectralPositionalVectorArray,
    width: na.AbstractSpectralPositionalVectorArray,
):
    result = ctis.scenes.gaussians(
        inputs=inputs,
        width=width,
    )

    assert result.inputs is inputs
    assert result.shape == inputs.shape
