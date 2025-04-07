"""This module provides soundcard-based audio output functionality for pamiq-
io."""

import logging
from typing import override

import numpy as np
import soundcard as sc
from numpy.typing import NDArray

from .base import AudioOutput


class SoundcardAudioOutput(AudioOutput):
    """Audio output implementation using the Soundcard library.

    This class outputs audio using the Soundcard library which provides
    cross-platform audio output capabilities.

    Examples:
        >>> audio_output = SoundcardAudioOutput(
        ...     sample_rate=44100,
        ...     device_id=None,  # Uses default output device
        ...     block_size=1024,
        ...     channels=2
        ... )
        >>> audio_frames = np.zeros((1024, 2), dtype=np.float32)  # Silence
        >>> audio_output.write(audio_frames)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        device_id: str | None = None,
        block_size: int | None = None,
        channels: int = 2,
    ) -> None:
        """Initializes an instance of SoundcardAudioOutput.

        Args:
            sample_rate: The desired sample rate in Hz.
            device_id: The audio output device id to use. Can be device name
                or None for default device.
            block_size: Size of each audio block for the player.
            channels: Number of audio channels to output (1 for mono, 2 for stereo).

        Raises:
            RuntimeError: If the specified device is not found or cannot be accessed.
        """
        # Get the speaker device
        try:
            if device_id is None:
                self._speaker = sc.default_speaker()
            else:
                self._speaker = sc.get_speaker(device_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio device: {e}") from e

        self._sample_rate = sample_rate
        self._channels = channels

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Open the playback stream
        self._stream = self._speaker.player(
            samplerate=sample_rate, channels=channels, blocksize=block_size
        )
        self._stream.__enter__()

        self.logger.debug(
            f"Initialized audio output with sample_rate={sample_rate}, "
            f"channels={channels}, block_size={block_size}"
        )

    @property
    @override
    def sample_rate(self) -> float:
        """Get the current sample rate of the audio output.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    @property
    @override
    def channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            The number of audio channels (1 for mono, 2 for stereo, etc.).
        """
        return self._channels

    @override
    def write(self, data: NDArray[np.float32]) -> None:
        """Writes audio frames to the output stream.

        Args:
            data: Audio data as a numpy array with shape (frame_size, channels)
                and values normalized between -1.0 and 1.0.

        Raises:
            RuntimeError: If the audio frames cannot be written.
            ValueError: If the data shape is incompatible with the configured channels.
        """
        try:
            # Ensure data is float32
            data = np.asarray(data, dtype=np.float32)

            # Check shape compatibility
            if data.ndim == 1:
                # Single channel data, reshape to (frames, 1)
                data = data.reshape(-1, 1)

            if data.ndim != 2:
                raise ValueError(f"Data must be 2D array, got shape {data.shape}")

            if data.shape[1] != self.channels:
                raise ValueError(
                    f"Data has {data.shape[1]} channels, but output configured for {self.channels} channels"
                )

            # Play the data
            self._stream.play(data)
        except Exception as e:
            self.logger.error(f"Error writing audio frames: {e}")
            raise RuntimeError(f"Failed to write audio frames: {e}") from e

    def __del__(self) -> None:
        """Cleanup method to properly close the audio stream when the object is
        destroyed."""
        if hasattr(self, "_stream"):
            try:
                self._stream.__exit__(None, None, None)
                self.logger.debug("Audio stream closed")
            except Exception:
                self.logger.exception("Error closing audio stream.")
                raise
