"""Tests for video_capture module."""

import cv2
import numpy as np
import pytest

from pamiq_io.vision.capture.opencv import OpenCVVideoCapture


class TestOpenCVVideoCapture:
    """Tests for OpenCVVideoCapture class."""

    def test_init_with_camera_index(self, mocker):
        """Test initialization with camera index."""
        mock_camera = mocker.patch("cv2.VideoCapture")
        mock_camera.return_value.set.return_value = True
        mock_camera.return_value.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
        }[prop]

        capture = OpenCVVideoCapture(camera=0)

        mock_camera.assert_called_once_with(index=0)
        assert capture.width == 640
        assert capture.height == 480
        assert capture.fps == 30
        assert capture.channels == 3  # Default value

    def test_init_with_camera_object(self, mocker):
        """Test initialization with camera object."""
        mock_camera = mocker.MagicMock()
        mock_camera.set.return_value = True
        mock_camera.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 60,
        }[prop]

        capture = OpenCVVideoCapture(
            camera=mock_camera, width=1280, height=720, fps=60, channels=4
        )

        assert capture.width == 1280
        assert capture.height == 720
        assert capture.fps == 60
        assert capture.channels == 4  # Custom channels value

    def test_configure_camera_warning_on_failure(self, mocker, caplog):
        """Test warning when camera config fails."""
        mock_camera = mocker.MagicMock()
        # Return False for set to simulate failure
        mock_camera.set.return_value = False
        # Return different values for get to simulate mismatch
        mock_camera.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 320,  # Different from expected 640
            cv2.CAP_PROP_FRAME_HEIGHT: 240,  # Different from expected 480
            cv2.CAP_PROP_FPS: 15,  # Different from expected 30
        }[prop]

        OpenCVVideoCapture(camera=mock_camera)

        # Check if warnings were logged
        assert "Failed to set width" in caplog.text
        assert "Failed to set height" in caplog.text
        assert "Failed to set fps" in caplog.text

    def test_read_success(self, mocker):
        """Test successful frame read."""
        mock_camera = mocker.MagicMock()
        # Create a fake frame with recognizable pattern
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Make read return a 3-channel frame
        mock_camera.read.return_value = (True, mock_frame)

        capture = OpenCVVideoCapture(camera=mock_camera)
        result = capture.read()

        assert result.shape == (480, 640, 3)

    def test_read_grayscale_success(self, mocker):
        """Test successful frame read for grayscale images."""
        mock_camera = mocker.MagicMock()
        # Create a grayscale frame (2D)
        mock_frame = np.zeros((480, 640), dtype=np.uint8)

        mock_camera.read.return_value = (True, mock_frame)

        # Set to expect 1 channel
        capture = OpenCVVideoCapture(camera=mock_camera, channels=1)
        result = capture.read()

        # Check that shape is (height, width, 1) after processing
        assert result.shape == (480, 640, 1)

    def test_read_channel_mismatch_error(self, mocker):
        """Test channel mismatch error during frame read."""
        mock_camera = mocker.MagicMock()
        # Create a frame with 3 channels
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_camera.read.return_value = (True, mock_frame)

        # Set to expect 1 channel, which doesn't match the frame
        capture = OpenCVVideoCapture(camera=mock_camera, channels=1)

        with pytest.raises(
            ValueError, match=r"Captured frame has 3 channels, but expected 1 channels"
        ):
            capture.read()

    def test_read_failure(self, mocker, caplog):
        """Test read failure after multiple attempts."""
        mock_camera = mocker.MagicMock()
        mock_camera.read.return_value = (False, None)  # Always fail

        capture = OpenCVVideoCapture(camera=mock_camera, num_trials_on_read_failure=3)

        with pytest.raises(RuntimeError, match="Failed to read capture frame"):
            capture.read()

        assert mock_camera.read.call_count == 3
        # Check that debug messages were logged for each retry
        assert "Failed to read capture frame, retrying (1/3)" in caplog.text
        assert "Failed to read capture frame, retrying (2/3)" in caplog.text
        assert "Failed to read capture frame, retrying (3/3)" in caplog.text
