import ctis
import named_arrays as na
from .._inverters_test import AbstractTestAbstractInverter


class AbstractTestAbstractIterativeInverter(
    AbstractTestAbstractInverter,
):

    def test__call__(
        self,
        a: ctis.inverters.AbstractIterativeInverter,
        images: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
        **kwargs,
    ) -> ctis.inverters.IterativeInversionResult:

        result = super().test__call__(
            a=a,
            images=images,
            **kwargs,
        )

        axis_iteration = result.inverter.axis_iteration

        assert result.iteration.size == result.num_iteration
        assert result.mean_chi_squared.shape[axis_iteration] == result.num_iteration
        assert result.correlation_residual.shape[axis_iteration] == result.num_iteration

        return result

    def test_num_iteration(self, a: ctis.inverters.AbstractIterativeInverter):
        result = a.num_iteration
        assert isinstance(result, int)
        assert result > 0
