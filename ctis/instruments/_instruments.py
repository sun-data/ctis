from typing import Callable, Sequence
import abc
import dataclasses
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractInstrument",
    "AbstractLinearInstrument",
    "IdealInstrument",
]

@dataclasses.dataclass
class AbstractCTIS:

    weights
    coordinates_scene: na.SpectralPositionalVectorArray
    coordinates_detector: na.SpectralPositionalVectorArray

    def project(self):

    def deproject(self):

@dataclasses.dataclass
class IdealCTIS(
    optika.systems._interpolated.AbstractInterpolatedSystem,
):
    """
    An interface describing an Ideal CTIS instrument with a linear distortion model, parameterized by dispersion angle,
    spatial and spectral plate scale, and a reference position on the detector (???)
    """

    angle: u.Quantity | na.AbstractScalar
    spatial_plate_scale: u.Quantity | na.AbstractScalar
    spectral_plate_scale: u.Quantity | na.AbstractScalar
    ref_position: u.Quantity | na.AbstractScalar

    def distortion(self, coordinates: na.SpectralPositionalVectorArray):
        delta_lambda = self.spectral_plate_scale / self.spatial_plate_scale
        rot = na.Cartesian2dRotationMatrixArray(self.angle)
        rot_grid = na.SpectralPositionalVectorArray(
            wavelength=coordinates.wavelength,
            position=rot @ coordinates.position
        )
        disperse = na.SpectralPositionalMatrixArray(
            wavelength=na.SpectralPositionalVectorArray(
                wavelength=1,
                position=na.Cartesian2dVectorArray(
                    x=0 * u.angstrom / u.arcsec,
                    y=0 * u.angstrom / u.arcsec,
                )
            ),
            position=na.Cartesian2dMatrixArray(
                x=na.SpectralPositionalVectorArray(
                    wavelength=1 / delta_lambda,
                    position=na.Cartesian2dVectorArray(
                        # originally I had this as x = -1 which resulted in the grid not being in ascending order.  This caused the interpolator to puke.
                        x=1,
                        y=0,
                    )
                ),
                y=na.SpectralPositionalVectorArray(
                    wavelength=0 * u.arcsec / u.angstrom,
                    position=na.Cartesian2dVectorArray(
                        x=0,
                        y=1,
                    )
                )
            )
        )
        projected_grid = disperse @ rot_grid
        projected_grid = projected_grid - self.ref_position
        return na.SpectralPositionalVectorArray(
            wavelength=coordinates.wavelength,
            position=projected_grid.position,
        )

    def weights(
            self,
            coordinates: na.SpectralPositionalVectorArray,
            axis_wavelength: str,
            axis_field: tuple[str, str],
    ) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        Compute the weights which map the overlap of each pixel on the object
        plane to each pixel on the detector plane.

        Parameters
        ----------
        coordinates
            The vertices of each pixel on the object plane.
        axis_wavelength
            The logical axis
        axis_field
            The logical axes corresponding to changing field coordinate.
        """

        position = coordinates.position

        position_new = self.distortion(coordinates)

        result = na.regridding.weights(
            coordinates_input=position,
            coordinates_output=position_new,
            axis_input=axis_field,
            axis_output=axis_field,
            method="conservative",
        )

        return result

    @abc.abstractmethod
    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        f"""
        The forward model of this CTIS instrument, which maps spectral radiance
        on the skyplane to counts on the detectors.
        
        Parameters
        ----------
        scene
            The spectral radiance in units equivalent to 
            {(u.erg / (u.cm**2 * u.sr * u.AA * u.s)):latex_inline}.
        """

        weights = self.weights(coordinates)
        regrid_from_weights

        return FunctionArray()



@dataclasses.dataclass
class CTIS:
    weights
    coordinates_input
    coordinates_output
    :


