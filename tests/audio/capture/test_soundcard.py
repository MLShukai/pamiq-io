"""Tests for the SoundcardAudioCapture class."""

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

from pamiq_io.audio.capture.soundcard import SoundcardAudioCapture


class TestSoundcardAudioCapture:
    """Tests for the SoundcardAudioCapture class."""

    @pytest.fixture
    def mock_mic(self, mocker: MockerFixture):
        """Creates a mock Microphone object."""
        mic = mocker.MagicMock()
        recorder = mocker.MagicMock()
        mic.recorder.return_value = recorder
        return mic

    @pytest.fixture
    def mock_sc(self, mocker: MockerFixture, mock_mic):
        """Mocks the soundcard module and returns the mock setup."""
        mock_sc = mocker.patch("pamiq_io.audio.capture.soundcard.sc")
        mock_sc.default_microphone.return_value = mock_mic
        mock_sc.get_microphone.return_value = mock_mic
        return mock_sc

    def test_init_default_device(self, mock_sc, mock_mic):
        """Tests initialization with default device."""

        SoundcardAudioCapture(
            sample_rate=44100, frame_size=1024, block_size=512, channels=1
        )

        # Check if default microphone was used
        mock_sc.default_microphone.assert_called_once()
        mock_mic.recorder.assert_called_once_with(
            samplerate=44100, channels=1, blocksize=512
        )
        # Verify stream was started
        mock_mic.recorder.return_value.__enter__.assert_called_once()

    def test_init_specific_device(self, mock_sc, mock_mic):
        """Tests initialization with a specific device ID."""

        SoundcardAudioCapture(
            sample_rate=48000, device_id="test_device", frame_size=2048, channels=2
        )

        # Check if specified device was used
        mock_sc.get_microphone.assert_called_once_with(
            "test_device", include_loopback=True
        )
        # Verify recorder was created with correct parameters
        mock_mic.recorder.assert_called_once_with(
            samplerate=48000, channels=2, blocksize=2048
        )

    def test_block_size_defaults_to_frame_size(self, mock_sc, mock_mic):
        """Tests that block_size defaults to frame_size when not specified."""

        SoundcardAudioCapture(sample_rate=44100, frame_size=1024)

        # Verify that blocksize is set to frame_size
        mock_mic.recorder.assert_called_once_with(
            samplerate=44100, channels=1, blocksize=1024
        )

    def test_property_getters(self, mock_sc):
        """Tests the property getter methods."""

        sample_rate = 48000
        frame_size = 2048
        channels = 2

        capture = SoundcardAudioCapture(
            sample_rate=sample_rate,
            frame_size=frame_size,
            channels=channels,
        )

        # Test property getters
        assert capture.sample_rate == sample_rate
        assert capture.frame_size == frame_size
        assert capture.channels == channels

    def test_read_success(self, mock_sc, mock_mic):
        """Tests successful reading of audio frames."""

        # Create test audio data
        test_frames = 1024
        test_channels = 2
        test_audio = np.random.uniform(-1.0, 1.0, (test_frames, test_channels)).astype(
            np.float32
        )

        # Configure mock to return test data
        recorder = mock_mic.recorder.return_value
        recorder.record.return_value = test_audio

        # Initialize audio capture
        capture = SoundcardAudioCapture(
            sample_rate=44100,
            frame_size=test_frames,
            channels=test_channels,
        )

        # Read audio frames
        result = capture.read()

        # Verify record was called with correct parameters
        recorder.record.assert_called_once_with(numframes=test_frames)

        # Check returned data
        assert np.array_equal(result, test_audio)
        assert result.dtype == np.float32
        assert result.shape == (test_frames, test_channels)

    def test_read_error(self, mock_sc, mock_mic, caplog):
        """Tests handling of errors during read operation."""

        # Configure the recorder mock to raise an exception
        recorder = mock_mic.recorder.return_value
        error_message = "Simulated audio read error"
        recorder.record.side_effect = Exception(error_message)

        capture = SoundcardAudioCapture(sample_rate=44100, frame_size=1024)

        # Attempt to read should raise RuntimeError
        with pytest.raises(
            RuntimeError, match=f"Failed to read audio frames:.*{error_message}"
        ):
            capture.read()

        # Check that error was logged
        assert "Error reading audio frames:" in caplog.text

    def test_cleanup_on_deletion(self, mock_sc, mock_mic, mocker: MockerFixture):
        """Tests that stream is properly closed on object deletion."""
        recorder = mock_mic.recorder.return_value

        # Create and then delete the capture object
        capture = SoundcardAudioCapture()

        # Use mocker to spy on __exit__ method
        exit_spy = mocker.spy(recorder, "__exit__")
        capture.__del__()

        # Verify that the stream was closed properly
        exit_spy.assert_called_once_with(None, None, None)

    def test_cleanup_handles_errors(
        self, mock_sc, mock_mic, caplog, mocker: MockerFixture
    ):
        """Tests that errors during cleanup are handled properly."""
        recorder = mock_mic.recorder.return_value

        # Configure exit to raise an exception
        exit_mock = mocker.patch.object(
            recorder, "__exit__", side_effect=Exception("Stream close error")
        )

        # Create capture object
        capture = SoundcardAudioCapture()

        # Deletion should raise the exception
        with pytest.raises(Exception, match="Stream close error"):
            capture.__del__()

        # Check that error was logged
        assert "Error closing audio stream" in caplog.text
        exit_mock.assert_called_once()
