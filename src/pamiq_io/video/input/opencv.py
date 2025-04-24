"""This module provides OpenCV-based video input implementation."""

import logging
from typing import TypedDict, override

import cv2
import numpy as np
from numpy.typing import NDArray

from .base import VideoInput


class DeviceInfo(TypedDict):
    index: int
    name: str
    resolution: tuple[int, int]


def list_video_devices(max_devices: int = 10) -> list[DeviceInfo]:
    """List all available video capture devices.

    Scans for video capture devices by attempting to open each index from 0 to max_devices-1.
    For each successfully opened device, retrieves basic information such as resolution and name.

    Args:
        max_devices: Maximum number of device indices to check.

    Returns:
        A list of dictionaries, each containing device information:
            - 'index': The device index
            - 'name': The device name (if available, otherwise empty string)
            - 'resolution': Tuple of (width, height)

    Examples:
        >>> devices = list_video_devices()
        >>> for device in devices:
        ...     print(f"Index: {device['index']}, Resolution: {device['resolution']}")
    """
    available_devices: list[DeviceInfo] = []
    logger = logging.getLogger(__name__)

    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get device information
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Attempt to get device name
            # Note: OpenCV may not provide device names on all platforms
            device_name = ""
            # This property might not be available on all platforms/OpenCV builds
            device_name = cap.getBackendName()

            device_info = DeviceInfo(
                index=i, name=device_name, resolution=(width, height)
            )
            available_devices.append(device_info)
            logger.debug(f"Found video device at index {i}: {width}x{height}")

        # Always release the capture object
        cap.release()

    return available_devices


def show_video_devices(max_devices: int = 10) -> None:
    """Display all available video capture devices.

    Lists all available video devices with their index, name (if available), and resolution.

    Args:
        max_devices: Maximum number of device indices to check.

    Examples:
        >>> show_video_devices()
        Available Video Capture Devices:
        [0] Resolution: 1280x720
        [1] Resolution: 640x480
    """
    devices = list_video_devices(max_devices)

    print("Available Video Capture Devices:")
    print("-------------------------------")

    if not devices:
        print("No video capture devices found.")
        return

    for device in devices:
        width, height = device["resolution"]
        device_name = f" - {device['name']}" if device["name"] else ""
        print(f"[{device['index']}]{device_name}, Resolution: {width}x{height}")


class OpenCVVideoInput(VideoInput):
    """Video input implementation using OpenCV.

    Attributes:
        camera: OpenCV VideoCapture.
        num_trials_on_read_failure: Number of trials on read failure.
        expected_width: Expected width of captured frame.
        expected_height: Expected height of captured frame.
        expected_fps: Expected FPS of capture.
        expected_channels: Expected number of channels in captured frame.

    Examples:
        >>> cam = OpenCVVideoInput(
        ... camera = cv2.VideoCapture(0),  # Use default camera
        ... width = 1280,
        ... height = 720,
        ... fps = 30,
        ... channels = 3,
        ... )
        >>> frame = cam.read()
    """

    def __init__(
        self,
        camera: cv2.VideoCapture | int,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        channels: int = 3,
        num_trials_on_read_failure: int = 10,
    ) -> None:
        """Initializes an instance of OpenCVVideoInput.

        Args:
            camera: The OpenCV VideoCapture object or camera index to use.
            width: The desired width of the video frames. If None, use the camera's default width.
            height: The desired height of the video frames. If None, use the camera's default height.
            fps: The desired frames per second (fps) of the video. If None, use the camera's default fps.
            channels: The desired number of color channels (default is 3 for RGB/BGR).
            num_trials_on_read_failure: Number of trials on read failure.
        """
        if isinstance(camera, int):
            camera = cv2.VideoCapture(index=camera)

        if not camera.isOpened():
            raise RuntimeError("Can not open camera.")

        self.camera = camera
        self.num_trials_on_read_failure = num_trials_on_read_failure

        self.expected_width = width
        self.expected_height = height
        self.expected_fps = fps
        self.expected_channels = channels

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.configure_camera()

    def configure_camera(self) -> None:
        """Configures the camera settings with the desired properties."""
        if self.expected_width is not None:
            if (
                not self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.expected_width)
                or self.width != self.expected_width
            ):
                self.logger.warning(f"Failed to set width to {self.expected_width}.")

        if self.expected_height is not None:
            if (
                not self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.expected_height)
                or self.height != self.expected_height
            ):
                self.logger.warning(f"Failed to set height to {self.expected_height}.")

        if self.expected_fps is not None:
            if (
                not self.camera.set(cv2.CAP_PROP_FPS, self.expected_fps)
                or self.fps != self.expected_fps
            ):
                self.logger.warning(f"Failed to set fps to {self.expected_fps}.")

    @property
    @override
    def channels(self) -> int:
        """Get the expected number of color channels for the video frames.

        Returns:
            The number of color channels for the video frames.
        """
        return self.expected_channels

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
        """Reads a frame from the video input.

        Returns:
            The frame read from the video input with shape (height, width, channels).

        Raises:
            RuntimeError: If the frame cannot be read after num_trials_on_read_failure attempts.
            ValueError: If the captured frame's number of channels doesn't match the expected channels.
        """
        for i in range(self.num_trials_on_read_failure):
            ret, frame = self.camera.read()
            if ret:
                # If the frame is grayscale (2D), add a channel dimension
                if frame.ndim == 2:
                    frame = np.expand_dims(frame, -1)

                # Verify that the frame has the expected number of channels
                if frame.shape[-1] != self.expected_channels:
                    raise ValueError(
                        f"Captured frame has {frame.shape[-1]} channels, but expected {self.expected_channels} channels."
                    )

                # Convert BGR to RGB
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.shape[-1] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

                return np.asarray(frame, dtype=np.uint8, copy=False)
            else:
                self.logger.warning(
                    f"Failed to read input frame, retrying ({i+1}/{self.num_trials_on_read_failure})..."
                )

        raise RuntimeError("Failed to read input frame.")
