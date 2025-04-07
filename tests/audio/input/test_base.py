import pytest

from pamiq_io.audio.input.base import AudioInput


class TestAudioInput:
    """Tests for the AudioInput abstract base class."""

    @pytest.mark.parametrize("method_name", ["read", "sample_rate", "channels"])
    def test_abstract_methods(self, method_name):
        """Test that AudioInput correctly defines expected abstract methods."""
        assert method_name in AudioInput.__abstractmethods__
