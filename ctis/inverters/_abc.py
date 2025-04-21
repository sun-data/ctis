import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractInverter",
]


@dataclasses.dataclass
class AbstractInverter(
    abc.ABC,
):
    """
    An interface describing an algorithm which can invert CTIS observations
    to yield a reconstruction of the observed scene.
    """

    def __call__(
        self,
        observation: na.FunctionArray[
            na.SpectralPositionalVectorArray,
            na.AbstractScalarArray,
        ],
        **kwargs,
    ):
        """
        Invert the given image deprojections to reconstruct the original scene.

        Parameters
        ----------
        observation
            Observed images which have been projected backwards through the scene.
            This defines the field of view and wavelength range of the
            reconstructed scene.
            We choose the deprojections instead of the images here since the
            former defines the extent of the reconstructed scene without the
            need for additional arguments.
        kwargs
            Additional keyword arguments which can be used by subclass
            implementations.
        """
