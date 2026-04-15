import abc
import numpy as np
import named_arrays as na
import ctis


class AbstractTestAbstractInverter(
    abc.ABC,
):

    def test_instrument(self, a: ctis.inverters.AbstractInverter):
        result = a.instrument
        assert isinstance(result, ctis.instruments.AbstractInstrument)

    def test__call__(
        self,
        a: ctis.inverters.AbstractInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    ):
        result = a(images)

        assert isinstance(result, ctis.inverters.InversionResult)

        images_new = a.instrument.image(result.solution)

        assert np.isclose(images, images_new)
