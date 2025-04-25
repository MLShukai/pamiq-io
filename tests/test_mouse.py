"""Tests for the mouse_output module."""

import time

import pytest

from pamiq_io.mouse import InputtinoMouseOutput, MouseButton


class TestInputtinoMouseOutput:
    """Tests for the InputtinoMouseOutput class."""

    @pytest.fixture
    def mock_mouse(self, mocker):
        """Create a mock for the Mouse class."""
        mock_instance = mocker.Mock()
        mocker.patch("pamiq_io.mouse.Mouse", return_value=mock_instance)
        return mock_instance

    def test_convert_to_mouse_button_with_string(self):
        """Test converting string literals to MouseButton enum values."""
        # Test all valid string literals
        assert InputtinoMouseOutput.convert_to_mouse_button("left") == MouseButton.LEFT
        assert (
            InputtinoMouseOutput.convert_to_mouse_button("right") == MouseButton.RIGHT
        )
        assert (
            InputtinoMouseOutput.convert_to_mouse_button("middle") == MouseButton.MIDDLE
        )
        assert InputtinoMouseOutput.convert_to_mouse_button("side") == MouseButton.SIDE
        assert (
            InputtinoMouseOutput.convert_to_mouse_button("extra") == MouseButton.EXTRA
        )

    def test_convert_to_mouse_button_with_enum(self):
        """Test that enum values pass through the converter unchanged."""
        # Test with direct enum values
        assert (
            InputtinoMouseOutput.convert_to_mouse_button(MouseButton.LEFT)
            == MouseButton.LEFT
        )
        assert (
            InputtinoMouseOutput.convert_to_mouse_button(MouseButton.RIGHT)
            == MouseButton.RIGHT
        )

    def test_press_with_string(self, mock_mouse):
        """Test pressing a mouse button using a string literal."""
        mouse_output = InputtinoMouseOutput()

        # Test with string literal
        mouse_output.press("left")
        mock_mouse.press.assert_called_once_with(MouseButton.LEFT)

    def test_press_with_enum(self, mock_mouse):
        """Test pressing a mouse button using MouseButton enum."""
        mouse_output = InputtinoMouseOutput()

        # Test with enum value
        mouse_output.press(MouseButton.RIGHT)
        mock_mouse.press.assert_called_once_with(MouseButton.RIGHT)

    def test_release_with_string(self, mock_mouse):
        """Test releasing a mouse button using a string literal."""
        mouse_output = InputtinoMouseOutput()

        # Test with string literal
        mouse_output.release("middle")
        mock_mouse.release.assert_called_once_with(MouseButton.MIDDLE)

    def test_release_with_enum(self, mock_mouse):
        """Test releasing a mouse button using MouseButton enum."""
        mouse_output = InputtinoMouseOutput()

        # Test with enum value
        mouse_output.release(MouseButton.SIDE)
        mock_mouse.release.assert_called_once_with(MouseButton.SIDE)

    def test_move(self, mock_mouse):
        """Test that move sets velocity correctly and the update loop moves the
        mouse."""
        # Create mouse with very high FPS to reduce test time
        mouse_output = InputtinoMouseOutput(fps=100.0)

        # Set velocity to 1000 pixels/second in both directions
        mouse_output.move(1000.0, 1000.0)

        # Wait a short time to ensure the update loop has run at least once
        time.sleep(0.05)

        # Check that the mouse was moved at least once
        mock_mouse.move.assert_called()

    def test_cleanup(self, mock_mouse):
        """Test that mouse is properly cleaned up on destruction."""
        # Create and immediately destroy
        mouse_output = InputtinoMouseOutput(fps=1000.0)
        mouse_output.__del__()
