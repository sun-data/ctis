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

        assert result.solution.sum() > 0
        assert result.success
        assert isinstance(result.message, str)
        assert np.all(result.images == images)
        assert result.inverter == a
