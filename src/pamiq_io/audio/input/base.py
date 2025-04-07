"""This module provides audio input functionality for game-io."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class AudioInput(ABC):
    """Abstract base class for audio input.

    This class defines the interface for audio input implementations.
    """

    @abstractmethod
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

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Get the current sample rate of the audio input.

        Returns:
            The sample rate in Hz.
        """
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        """Get the number of audio channels.

        Returns:
            The number of audio channels (1 for mono, 2 for stereo, etc.).
        """
        ...
