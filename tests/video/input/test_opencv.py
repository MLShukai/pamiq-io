"""Tests for video_input module."""

import cv2
import numpy as np
import pytest

from pamiq_io.video.input.opencv import OpenCVVideoInput


class TestOpenCVVideoInput:
    """Tests for OpenCVVideoInput class."""

    def test_init_with_camera_index(self, mocker):
        """Test initialization with camera index."""
        mock_camera = mocker.patch("cv2.VideoCapture")
        mock_camera.return_value.set.return_value = True
        mock_camera.return_value.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
        }[prop]

        capture = OpenCVVideoInput(camera=0)

        mock_camera.assert_called_once_with(index=0)
        assert capture.width == 640
        assert capture.height == 480
        assert capture.fps == 30
        assert capture.channels == 3  # Default value

    def test_cannot_open_camera(self, mocker):
        mock_camera = mocker.patch("cv2.VideoCapture")()
        mock_camera.isOpened.return_value = False
        with pytest.raises(RuntimeError, match="Can not open camera."):
            OpenCVVideoInput(0)

        with pytest.raises(RuntimeError, match="Can not open camera."):
            OpenCVVideoInput(mock_camera)

    def test_init_with_camera_object(self, mocker):
        """Test initialization with camera object."""
        mock_camera = mocker.MagicMock()
        mock_camera.set.return_value = True
        mock_camera.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 60,
        }[prop]

        capture = OpenCVVideoInput(
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

        OpenCVVideoInput(camera=mock_camera, width=1, height=1, fps=1)

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
        capture = OpenCVVideoInput(camera=mock_camera)

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
        capture = OpenCVVideoInput(camera=mock_camera, channels=4)

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
        capture = OpenCVVideoInput(camera=mock_camera, channels=1)
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
        capture = OpenCVVideoInput(camera=mock_camera, channels=1)

        with pytest.raises(
            ValueError, match=r"Captured frame has 3 channels, but expected 1 channels"
        ):
            capture.read()

    def test_read_failure(self, mocker, caplog):
        """Test read failure after multiple attempts."""
        mock_camera = mocker.MagicMock()
        mock_camera.read.return_value = (False, None)  # Always fail

        capture = OpenCVVideoInput(camera=mock_camera, num_trials_on_read_failure=3)

        with pytest.raises(RuntimeError, match="Failed to read input frame"):
            capture.read()

        assert mock_camera.read.call_count == 3
        # Check that debug messages were logged for each retry
        assert "Failed to read input frame, retrying (1/3)" in caplog.text
        assert "Failed to read input frame, retrying (2/3)" in caplog.text
        assert "Failed to read input frame, retrying (3/3)" in caplog.text

    def test_init_with_none_parameters(self, mocker):
        """Test initialization with None parameters."""
        mock_camera = mocker.patch("cv2.VideoCapture")
        mock_camera.return_value.set.return_value = True
        mock_camera.return_value.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30,
        }[prop]

        # Initialize with None parameters
        capture = OpenCVVideoInput(camera=0, width=None, height=None, fps=None)

        # Verify camera is created
        mock_camera.assert_called_once_with(index=0)

        # Verify that set() was not called for the None parameters
        for call in mock_camera.return_value.set.call_args_list:
            args = call[0]
            assert args[0] not in [
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            ]

        # Verify default values are obtained from the camera
        assert capture.width == 1280
        assert capture.height == 720
        assert capture.fps == 30
        assert capture.channels == 3  # Default value

    def test_configure_camera_with_none_parameters(self, mocker, caplog):
        """Test configure_camera with None parameters."""
        mock_camera = mocker.patch("cv2.VideoCapture")

        # Return different values for get to simulate camera properties
        mock_camera.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30,
        }[prop]

        # Create capture with None parameters
        capture = OpenCVVideoInput(
            camera=mock_camera, width=None, height=None, fps=None
        )

        # Reset mock to clear any calls from initialization
        mock_camera.set.reset_mock()
        mock_camera.get.reset_mock()

        # Configure camera
        capture.configure_camera()

        # Verify set() was not called for None parameters
        mock_camera.set.assert_not_called()

        # Verify no warnings were logged
        for record in caplog.records:
            assert "Failed to set width" not in record.message
            assert "Failed to set height" not in record.message
            assert "Failed to set fps" not in record.message

    def test_configure_camera_with_mixed_parameters(self, mocker, caplog):
        """Test configure_camera with a mix of None and specified
        parameters."""
        mock_camera = mocker.MagicMock()

        # Return different values for get to simulate camera properties
        mock_camera.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30,
        }[prop]

        # Set() returns True for success
        mock_camera.set.return_value = True

        # Create capture with mixed parameters (width=None but height and fps specified)
        capture = OpenCVVideoInput(camera=mock_camera, width=None, height=480, fps=60)

        # Reset mock to clear any calls from initialization
        mock_camera.set.reset_mock()

        # Configure camera
        capture.configure_camera()

        # Verify set() was called only for non-None parameters
        assert mock_camera.set.call_count == 2
        mock_camera.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        mock_camera.set.assert_any_call(cv2.CAP_PROP_FPS, 60)

        # Verify that set() was NOT called for None parameters
        for call in mock_camera.set.call_args_list:
            args = call[0]
            assert args[0] != cv2.CAP_PROP_FRAME_WIDTH
