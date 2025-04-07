"""This module provides audio output functionality for pamiq-io."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class AudioOutput(ABC):
    """Abstract base class for audio output.

    This class defines the interface for audio output implementations.
    """

    @abstractmethod
    def write(self, data: NDArray[np.float32]) -> None:
        """Writes audio frames to the output stream.

        Args:
            data: Audio data as a numpy array with shape (frame_size, channels)
                and values normalized between -1.0 and 1.0.

        Raises:
            RuntimeError: If the audio frames cannot be written.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Get the current sample rate of the audio output.

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
