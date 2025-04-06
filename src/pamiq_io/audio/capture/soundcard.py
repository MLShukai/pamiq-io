"""This module provides audio capture functionality for game-io."""

import logging
from typing import override

import numpy as np
import soundcard as sc
from numpy.typing import NDArray

from .base import AudioCapture


class SoundcardAudioCapture(AudioCapture):
    """Audio capture implementation using the Soundcard library.

    This class captures audio using the Soundcard library which provides
    cross-platform audio capture capabilities.

    Examples:
        >>> audio_capture = SoundcardAudioCapture(
        ...     samplerate=44100,
        ...     device_id=None,  # Uses default input device
        ...     frame_size=1024,
        ...     blocksize=1024,
        ...     channels=1
        ... )
        >>> audio_frames = audio_capture.read()
    """

    def __init__(
        self,
        samplerate: int = 44100,
        device_id: str | None = None,
        frame_size: int = 1024,
        blocksize: int | None = None,
        channels: int = 1,
    ) -> None:
        """Initializes an instance of SoundcardAudioCapture.

        Args:
            samplerate: The desired sample rate in Hz.
            device_id: The audio input device id to use. Can be device name
                or None for default device.
            frame_size: Number of frames to read in each capture.
            blocksize: Size of each audio block for the recorder.
                If None, frame_size will be used.
            channels: Number of audio channels to capture (1 for mono, 2 for stereo).

        Raises:
            RuntimeError: If the specified device is not found or cannot be accessed.
        """
        # Get the microphone device
        try:
            if device_id is None:
                self._mic = sc.default_microphone()
            else:
                self._mic = sc.get_microphone(device_id, include_loopback=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio device: {e}") from e

        self._frame_size = frame_size
        self._blocksize = frame_size if blocksize is None else blocksize
        self._samplerate = samplerate
        self._channels = channels

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Open the recording stream
        try:
            self._stream = self._mic.recorder(
                samplerate=samplerate, channels=channels, blocksize=self._blocksize
            )
            self._stream.__enter__()
        except Exception as e:
            raise RuntimeError(f"Failed to start audio capture: {e}") from e

        self.logger.info(
            f"Initialized audio capture with samplerate={samplerate}, "
            f"channels={channels}, frame_size={frame_size}"
        )

    @property
    @override
    def sample_rate(self) -> float:
        """Get the current sample rate of the audio capture.

        Returns:
            The sample rate in Hz.
        """
        return self._samplerate

    @property
    @override
    def channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            The number of audio channels (1 for mono, 2 for stereo, etc.).
        """
        return self._channels

    @property
    @override
    def frame_size(self) -> int:
        """Get the frame size of the audio capture.

        Returns:
            The number of frames read in each capture.
        """
        return self._frame_size

    @override
    def read(self) -> NDArray[np.float32]:
        """Reads audio frames from the input stream.

        Returns:
            Audio data as a numpy array with shape (frame_size, channels)
            and values normalized between -1.0 and 1.0.

        Raises:
            RuntimeError: If the audio frames cannot be read.
        """
        try:
            frames = self._stream.record(numframes=self._frame_size)
            return np.asarray(frames, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error reading audio frames: {e}")
            raise RuntimeError(f"Failed to read audio frames: {e}") from e

    def __del__(self) -> None:
        """Cleanup method to properly close the audio stream when the object is
        destroyed."""
        if hasattr(self, "_stream"):
            try:
                self._stream.__exit__(None, None, None)
                self.logger.debug("Audio stream closed")
            except Exception as e:
                self.logger.warning(f"Error closing audio stream: {e}")
