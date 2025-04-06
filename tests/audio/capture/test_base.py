import pytest

from pamiq_io.audio.capture.base import AudioCapture


class TestAudioCapture:
    """Tests for the AudioCapture abstract base class."""

    @pytest.mark.parametrize(
        "method_name", ["read", "sample_rate", "channels", "frame_size"]
    )
    def test_abstract_methods(self, method_name):
        """Test that AudioCapture correctly defines expected abstract
        methods."""
        assert method_name in AudioCapture.__abstractmethods__
