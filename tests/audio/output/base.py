import pytest

from pamiq_io.audio.output.base import AudioOutput


class TestAudioOutput:
    """Tests for the AudioOutput abstract base class."""

    @pytest.mark.parametrize("method_name", ["write", "sample_rate", "channels"])
    def test_abstract_methods(self, method_name):
        """Test that AudioOutput correctly defines expected abstract
        methods."""
        assert method_name in AudioOutput.__abstractmethods__
