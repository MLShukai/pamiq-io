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

    def test_read_success_with_bgr_to_rgb_conversion(self, mocker):
        """Test successful frame read with BGR to RGB conversion."""
        mock_camera = mocker.MagicMock()

        # Create a simple BGR test pattern (Blue, Green, Red)
        # Blue pixel (255, 0, 0) in BGR should become (0, 0, 255) in RGB
        bgr_frame = np.zeros((1, 3, 3), dtype=np.uint8)
        # First pixel: Blue in BGR (255, 0, 0)
        bgr_frame[0, 0] = [255, 0, 0]
        # Second pixel: Green in BGR (0, 255, 0)
        bgr_frame[0, 1] = [0, 255, 0]
        # Third pixel: Red in BGR (0, 0, 255)
        bgr_frame[0, 2] = [0, 0, 255]

        # Set up the mock to return our test frame
        mock_camera.read.return_value = (True, bgr_frame)

        # Create capture object with mock camera
        capture = OpenCVVideoCapture(camera=mock_camera)

        # Get the frame with color conversion applied
        result = capture.read()

        # Verify color conversion: BGR to RGB
        # Blue in BGR (255, 0, 0) should be Red in RGB (255, 0, 0)
        assert result[0, 0, 0] == 0  # R channel (was B)
        assert result[0, 0, 1] == 0  # G channel
        assert result[0, 0, 2] == 255  # B channel (was R)

        # Green stays the same in BGR and RGB
        assert result[0, 1, 0] == 0  # R channel
        assert result[0, 1, 1] == 255  # G channel
        assert result[0, 1, 2] == 0  # B channel

        # Red in BGR (0, 0, 255) should be Blue in RGB (0, 0, 255)
        assert result[0, 2, 0] == 255  # R channel (was B)
        assert result[0, 2, 1] == 0  # G channel
        assert result[0, 2, 2] == 0  # B channel (was R)

    def test_read_with_bgra_to_rgba_conversion(self, mocker):
        """Test frame read with BGRA to RGBA conversion for 4-channel
        images."""
        mock_camera = mocker.MagicMock()

        # Create a BGRA test pattern with alpha
        bgra_frame = np.zeros((1, 2, 4), dtype=np.uint8)
        # First pixel: Blue with full opacity in BGRA (255, 0, 0, 255)
        bgra_frame[0, 0] = [255, 0, 0, 255]
        # Second pixel: Transparent red in BGRA (0, 0, 255, 128)
        bgra_frame[0, 1] = [0, 0, 255, 128]

        mock_camera.read.return_value = (True, bgra_frame)

        # Create capture object with 4 channels
        capture = OpenCVVideoCapture(camera=mock_camera, channels=4)

        # Get the frame with color conversion applied
        result = capture.read()

        # Verify BGRA to RGBA conversion
        # Blue in BGRA should become Red in RGBA with preserved alpha
        assert result[0, 0, 0] == 0  # R channel (was B)
        assert result[0, 0, 1] == 0  # G channel
        assert result[0, 0, 2] == 255  # B channel (was R)
        assert result[0, 0, 3] == 255  # Alpha unchanged

        # Red in BGRA should become Blue in RGBA with preserved alpha
        assert result[0, 1, 0] == 255  # R channel (was B)
        assert result[0, 1, 1] == 0  # G channel
        assert result[0, 1, 2] == 0  # B channel (was R)
        assert result[0, 1, 3] == 128  # Alpha unchanged

    def test_read_grayscale_success(self, mocker):
        """Test successful frame read for grayscale images."""
        mock_camera = mocker.MagicMock()
        # Create a grayscale frame (2D)
        mock_frame = np.zeros((480, 640), dtype=np.uint8)
        # Add some values for testing
        mock_frame[240, 320] = 128

        mock_camera.read.return_value = (True, mock_frame)

        # Set to expect 1 channel
        capture = OpenCVVideoCapture(camera=mock_camera, channels=1)
        result = capture.read()

        # Check that shape is (height, width, 1) after processing
        assert result.shape == (480, 640, 1)
        # Verify values are preserved
        assert result[240, 320, 0] == 128

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
