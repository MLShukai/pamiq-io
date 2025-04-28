import pytest

from pamiq_io.keyboard.output.base import KeyboardOutput


class TestKeyboardOutput:
    @pytest.mark.parametrize("method", ["press", "release"])
    def test_abstract_method(self, method):
        assert method in KeyboardOutput.__abstractmethods__
