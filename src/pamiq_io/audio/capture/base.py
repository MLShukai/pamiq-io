"""This module provides audio capture functionality for game-io."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class AudioCapture(ABC):
    """Abstract base class for audio capture.

    This class defines the interface for audio capture implementations.
    """

    @abstractmethod
    def read(self) -> NDArray[np.float32]:
        """Reads audio frames from the input stream.

        Returns:
            Audio data as a numpy array with shape (frame_size, channels)
            and values normalized between -1.0 and 1.0.

        Raises:
            RuntimeError: If the audio frames cannot be read.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Get the current sample rate of the audio capture.

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

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Get the frame size of the audio capture.

        Returns:
            The number of frames read in each capture.
        """
        ...
