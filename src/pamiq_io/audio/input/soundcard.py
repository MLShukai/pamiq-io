"""This module provides audio input functionality for game-io."""

import logging
from typing import override

import numpy as np
import soundcard as sc
from numpy.typing import NDArray

from .base import AudioInput


def show_all_input_devices() -> None:
    """Display all available audio input devices.

    Lists all available audio input devices with their index, ID, and name.
    Includes loopback devices (virtual microphones that record speaker output).

    Examples:
        >>> show_all_input_devices()
        Available Audio Input Devices:
        [0] ID: "device1" - Built-in Microphone
        [1] ID: "device2" - External USB Microphone
        [2] ID: "device3" - System Audio (Loopback)
    """
    print("Default Microhpone Device:")
    print("------------------------------")
    mic = sc.default_microphone()
    print(f'[*] ID: "{mic.id}" - {mic.name}')
    print()
    print("Available Audio Input Devices:")
    print("------------------------------")

    for i, mic in enumerate(sc.all_microphones(include_loopback=True)):
        print(f'[{i}] ID: "{mic.id}" - {mic.name}')


class SoundcardAudioInput(AudioInput):
    """Audio input implementation using the Soundcard library.

    This class captures audio using the Soundcard library which provides
    cross-platform audio input capabilities.

    Examples:
        >>> audio_input = SoundcardAudioInput(
        ...     sample_rate=44100,
        ...     device_id=None,  # Uses default input device
        ...     block_size=1024,
        ...     channels=1
        ... )
        >>> audio_frames = audio_input.read(frame_size=1024)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        device_id: str | None = None,
        block_size: int | None = None,
        channels: int = 1,
    ) -> None:
        """Initializes an instance of SoundcardAudioInput.

        Args:
            sample_rate: The desired sample rate in Hz.
            device_id: The audio input device id to use. Can be device name
                or None for default device.
            block_size: Size of each audio block for the recorder.
            channels: Number of audio channels to input (1 for mono, 2 for stereo).
        """
        # Get the microphone device
        if device_id is None:
            self._mic = sc.default_microphone()
        else:
            self._mic = sc.get_microphone(device_id, include_loopback=True)

        self._sample_rate = sample_rate
        self._channels = channels

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Open the recording stream
        self._stream = self._mic.recorder(
            samplerate=sample_rate, channels=channels, blocksize=block_size
        )
        self._stream.__enter__()

        self.logger.debug(
            f"Initialized audio input with sample_rate={sample_rate}, "
            f"channels={channels}, block_size={block_size}"
        )

    @property
    @override
    def sample_rate(self) -> float:
        """Get the current sample rate of the audio input.

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
    def read(self, frame_size: int) -> NDArray[np.float32]:
        """Reads audio frames from the input stream.

        Args:
            frame_size: Number of frames to read.

        Returns:
            Audio data as a numpy array with shape (frame_size, channels)
            and values normalized between -1.0 and 1.0.

        Raises:
            RuntimeError: If the audio frames cannot be read.
        """
        frames = self._stream.record(numframes=frame_size)
        return np.asarray(frames, dtype=np.float32)

    def __del__(self) -> None:
        """Cleanup method to properly close the audio stream when the object is
        destroyed."""
        if hasattr(self, "_stream"):
            self._stream.__exit__(None, None, None)
            self.logger.debug("Audio stream closed")
