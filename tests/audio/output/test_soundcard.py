"""Tests for the SoundcardAudioOutput class."""

import shutil
import sys

import numpy as np
import pytest
from pytest_mock import MockerFixture

# Skip tests if required audio backends are not available on Linux
if sys.platform == "linux":
    if shutil.which("pipewire") is None and shutil.which("pulseaudio") is None:
        pytest.skip(
            "Linux audio backend system (pipewire or pulseaudio) is not available.",
            allow_module_level=True,
        )

from pamiq_io.audio.output.soundcard import SoundcardAudioOutput


class TestSoundcardAudioOutput:
    """Tests for the SoundcardAudioOutput class."""

    @pytest.fixture
    def mock_speaker(self, mocker: MockerFixture):
        """Creates a mock Speaker object."""
        speaker = mocker.MagicMock()
        player = mocker.MagicMock()
        speaker.player.return_value = player
        return speaker

    @pytest.fixture
    def mock_sc(self, mocker: MockerFixture, mock_speaker):
        """Mocks the soundcard module and returns the mock setup."""
        mock_sc = mocker.patch("pamiq_io.audio.output.soundcard.sc")
        mock_sc.default_speaker.return_value = mock_speaker
        mock_sc.get_speaker.return_value = mock_speaker
        return mock_sc

    def test_init_default_device(self, mock_sc, mock_speaker):
        """Tests initialization with default device."""

        SoundcardAudioOutput(sample_rate=44100, block_size=512, channels=2)

        # Check if default speaker was used
        mock_sc.default_speaker.assert_called_once()
        mock_speaker.player.assert_called_once_with(
            samplerate=44100, channels=2, blocksize=512
        )
        # Verify stream was started
        mock_speaker.player.return_value.__enter__.assert_called_once()

    def test_init_specific_device(self, mock_sc, mock_speaker):
        """Tests initialization with a specific device ID."""

        SoundcardAudioOutput(
            sample_rate=48000, device_id="test_device", block_size=2048, channels=2
        )

        # Check if specified device was used
        mock_sc.get_speaker.assert_called_once_with("test_device")
        # Verify player was created with correct parameters
        mock_speaker.player.assert_called_once_with(
            samplerate=48000, channels=2, blocksize=2048
        )

    def test_property_getters(self, mock_sc):
        """Tests the property getter methods."""

        sample_rate = 48000
        channels = 2

        output = SoundcardAudioOutput(
            sample_rate=sample_rate,
            channels=channels,
        )

        # Test property getters
        assert output.sample_rate == sample_rate
        assert output.channels == channels

    def test_write_success(self, mock_sc, mock_speaker):
        """Tests successful writing of audio frames."""

        # Create test audio data
        test_frames = 1024
        test_channels = 2
        test_audio = np.random.uniform(-1.0, 1.0, (test_frames, test_channels)).astype(
            np.float32
        )

        # Initialize audio output
        output = SoundcardAudioOutput(
            sample_rate=44100,
            channels=test_channels,
        )

        # Write audio frames
        output.write(test_audio)

        # Verify play was called with correct data
        player = mock_speaker.player.return_value
        player.play.assert_called_once()
        # Check that the data passed to play matches our test data
        np.testing.assert_array_equal(player.play.call_args[0][0], test_audio)

    def test_write_single_channel(self, mock_sc, mock_speaker):
        """Tests writing single channel data."""

        # Create single channel test audio data
        test_frames = 1024
        test_audio = np.random.uniform(-1.0, 1.0, test_frames).astype(np.float32)

        # Initialize audio output for single channel
        output = SoundcardAudioOutput(
            sample_rate=44100,
            channels=1,
        )

        # Write audio frames
        output.write(test_audio)

        # Verify play was called
        player = mock_speaker.player.return_value
        player.play.assert_called_once()
        # Check shape of data passed to play
        assert player.play.call_args[0][0].shape == (test_frames, 1)

    def test_write_channel_mismatch(self, mock_sc, mock_speaker):
        """Tests error handling for channel count mismatch."""

        # Create stereo audio data
        test_frames = 1024
        test_audio = np.random.uniform(-1.0, 1.0, (test_frames, 2)).astype(np.float32)

        # Initialize audio output for mono
        output = SoundcardAudioOutput(
            sample_rate=44100,
            channels=1,
        )

        # Writing stereo data to mono output should raise ValueError
        with pytest.raises(
            ValueError,
            match=r"Data has 2 channels, but output configured for 1 channels",
        ):
            output.write(test_audio)

    def test_cleanup_on_deletion(self, mock_sc, mock_speaker, mocker: MockerFixture):
        """Tests that stream is properly closed on object deletion."""
        player = mock_speaker.player.return_value

        # Create and then delete the output object
        output = SoundcardAudioOutput()

        # Use mocker to spy on __exit__ method
        exit_spy = mocker.spy(player, "__exit__")
        output.__del__()

        # Verify that the stream was closed properly
        exit_spy.assert_called_once_with(None, None, None)
