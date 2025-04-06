"""This module provides video capture functionality for game-io."""

import logging
from abc import ABC, abstractmethod
from typing import override

import cv2
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
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Get the current width of the video frames.

        Returns:
            The current width of the video frames.
        """
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Get the current height of the video frames.

        Returns:
            The current height of the video frames.
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Get the current frames per second (fps).

        Returns:
            The current frames per second (fps) of the video.
        """
        pass


class OpenCVVideoCapture(VideoCapture):
    """Video capture implementation using OpenCV.

    Attributes:
        camera: OpenCV VideoCapture.
        num_trials_on_read_failure: Number of trials on read failure.
        expected_width: Expected width of captured frame.
        expected_height: Expected height of captured frame.
        expected_fps: Expected FPS of capture.

    Examples:
        >>> cam = OpenCVVideoCapture(
        ... camera = cv2.VideoCapture(0),  # Use default camera
        ... width = 1280,
        ... height = 720,
        ... fps = 30,
        ... )
        >>> frame = cam.read()
    """

    def __init__(
        self,
        camera: cv2.VideoCapture | int,
        width: int = 640,
        height: int = 480,
        fps: float = 30,
        num_trials_on_read_failure: int = 10,
    ) -> None:
        """Initializes an instance of OpenCVVideoCapture.

        Args:
            camera: The OpenCV VideoCapture object or camera index to use.
            width: The desired width of the video frames.
            height: The desired height of the video frames.
            fps: The desired frames per second (fps) of the video.
            num_trials_on_read_failure: Number of trials on read failure.
        """
        if isinstance(camera, int):
            camera = cv2.VideoCapture(index=camera)

        self.camera = camera
        self.num_trials_on_read_failure = num_trials_on_read_failure

        self.expected_width = width
        self.expected_height = height
        self.expected_fps = fps

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.configure_camera()

    def configure_camera(self) -> None:
        """Configures the camera settings with the desired properties."""
        if (
            not self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.expected_width)
            or self.width != self.expected_width
        ):
            self.logger.warning(f"Failed to set width to {self.expected_width}.")
        if (
            not self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.expected_height)
            or self.height != self.expected_height
        ):
            self.logger.warning(f"Failed to set height to {self.expected_height}.")
        if (
            not self.camera.set(cv2.CAP_PROP_FPS, self.expected_fps)
            or self.fps != self.expected_fps
        ):
            self.logger.warning(f"Failed to set fps to {self.expected_fps}.")

    @property
    @override
    def width(self) -> int:
        """Get the current width of the video frames from the camera.

        Returns:
            The current width of the video frames.
        """
        return int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    @override
    def height(self) -> int:
        """Get the current height of the video frames from the camera.

        Returns:
            The current height of the video frames.
        """
        return int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    @override
    def fps(self) -> float:
        """Get the current frames per second (fps) from the camera.

        Returns:
            The current frames per second (fps) of the video.
        """
        return float(self.camera.get(cv2.CAP_PROP_FPS))

    @override
    def read(self) -> NDArray[np.uint8]:
        """Reads a frame from the video capture.

        Returns:
            The frame read from the video capture with shape (height, width, 3).

        Raises:
            RuntimeError: If the frame cannot be read after num_trials_on_read_failure attempts.
        """
        for i in range(self.num_trials_on_read_failure):
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB by default
                return np.asarray(frame, dtype=np.uint8, copy=False)
            else:
                self.logger.warning(
                    f"Failed to read capture frame, retrying ({i+1}/{self.num_trials_on_read_failure})..."
                )

        raise RuntimeError("Failed to read capture frame.")
