import pytest
import abc
import ctis


class AbstractTestAbstractInstrument(
    abc.ABC,
):
    def test_project(self, a: ctis.instruments.AbstractInstrument):
        result = a.project
        assert hasattr(result, "__call__")

    def test_deproject(self, a: ctis.instruments.AbstractInstrument):
        result = a.deproject
        assert hasattr(result, "__call__")


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        ctis.instruments.Instrument(
            project=lambda x: x,
            deproject=lambda x: x,
        )
    ]
)
class TestInstrument(
    abc.ABC,
):
    pass
