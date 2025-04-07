"""This module provides soundcard-based audio output functionality for pamiq-
io."""

import logging
from typing import override

import numpy as np
import soundcard as sc
from numpy.typing import NDArray

from .base import AudioOutput


def show_all_output_devices() -> None:
    """Display all available audio output devices.

    Lists all available audio output devices with their index, ID, and name.

    Examples:
        >>> show_all_output_devices()
        Available Audio Output Devices:
        [0] ID: "device1" - Built-in Speakers
        [1] ID: "device2" - HDMI Audio Output
        [2] ID: "device3" - External USB Headphones
    """
    print("Default Speaker Device:")
    print("-------------------------------")
    speaker = sc.default_speaker()
    print(f'[*] ID: "{speaker.id}" - {speaker.name}')

    print("Available Audio Output Devices:")
    print("-------------------------------")

    for i, speaker in enumerate(sc.all_speakers()):
        print(f'[{i}] ID: "{speaker.id}" - {speaker.name}')


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
        """
        # Get the speaker device
        if device_id is None:
            self._speaker = sc.default_speaker()
        else:
            self._speaker = sc.get_speaker(device_id)

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
            ValueError: If the data shape is incompatible with the configured channels.
        """
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

    def __del__(self) -> None:
        """Cleanup method to properly close the audio stream when the object is
        destroyed."""
        if hasattr(self, "_stream"):
            self._stream.__exit__(None, None, None)
            self.logger.debug("Audio stream closed")
