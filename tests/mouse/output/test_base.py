import pytest

from pamiq_io.mouse.output.base import MouseOutput


class TestMouseOutput:
    """Tests for the MouseOutput abstract base class."""

    @pytest.mark.parametrize("method", ["press", "release", "move"])
    def test_abstract_methods(self, method):
        assert method in MouseOutput.__abstractmethods__
