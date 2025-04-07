"""This module provides base classes for video capture functionality."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class VideoCapture(ABC):
    """Abstract base class for video capture.

    This class defines the interface for video capture implementations.
    """

    @abstractmethod
    def read(self) -> NDArray[np.uint8]:
        """Reads a frame from the video capture.

        Returns:
            The frame read from the video capture with shape (height, width, channels).

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
