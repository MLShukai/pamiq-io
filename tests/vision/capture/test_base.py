import pytest

from pamiq_io.vision.capture.base import VideoCapture


class TestVideoCapture:
    """Tests for the VideoCapture abstract base class."""

    @pytest.mark.parametrize(
        "method_name", ["read", "width", "height", "fps", "channels"]
    )
    def test_abstract_methods(self, method_name):
        """Test that VideoCapture correctly defines expected abstract
        methods."""
        assert method_name in VideoCapture.__abstractmethods__
