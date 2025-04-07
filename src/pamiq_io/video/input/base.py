"""This module provides base classes for video input functionality."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class VideoInput(ABC):
    """Abstract base class for video input.

    This class defines the interface for video input implementations.
    """

    @abstractmethod
    def read(self) -> NDArray[np.uint8]:
        """Reads a frame from the video input.

        Returns:
            The frame read from the video input with shape (height, width, channels).

        Raises:
            RuntimeError: If the frame cannot be read.
        """
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        """Get the current number of color channels for the video frames.

        Returns:
            The number of color channels (e.g., 1 for grayscale, 3 for RGB, 4 for RGBA).
        """
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        """Get the current width of the video frames.

        Returns:
            The current width of the video frames.
        """
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Get the current height of the video frames.

        Returns:
            The current height of the video frames.
        """
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        """Get the current frames per second (fps).

        Returns:
            The current frames per second (fps) of the video.
        """
        ...
