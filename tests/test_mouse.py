"""Tests for the mouse_output module."""

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

    def test_move(self, mock_mouse):
        """Test moving the mouse cursor."""
        mouse_output = InputtinoMouseOutput()
        mouse_output.move(100, -50)

        # Verify that move was called with the correct parameters
        mock_mouse.move.assert_called_once_with(100, -50)

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
