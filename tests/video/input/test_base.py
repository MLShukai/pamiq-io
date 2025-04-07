import pytest

from pamiq_io.video.input.base import VideoInput


class TestVideoInput:
    """Tests for the VideoInput abstract base class."""

    @pytest.mark.parametrize(
        "method_name", ["read", "width", "height", "fps", "channels"]
    )
    def test_abstract_methods(self, method_name):
        """Test that VideoInput correctly defines expected abstract methods."""
        assert method_name in VideoInput.__abstractmethods__
