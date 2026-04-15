import ctis
from .._inverters_test import AbstractTestAbstractInverter


class AbstractTestAbstractIterativeInverter(
    AbstractTestAbstractInverter,
):

    def test_num_iteration(self, a: ctis.inverters.AbstractIterativeInverter):
        result = a.num_iteration
        assert isinstance(result, int)
        assert  result > 0
