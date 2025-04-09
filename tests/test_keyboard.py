"""Tests for the keyboard_output module."""

import pytest

from pamiq_io.keyboard import InputtinoKeyboardOutput, KeyCode


class TestInputtinoKeyboardOutput:
    """Tests for the InputtinoKeyboardOutput class."""

    @pytest.fixture
    def mock_keyboard(self, mocker):
        """Create a mock for the Keyboard class."""
        mock_instance = mocker.MagicMock()
        mocker.patch("pamiq_io.keyboard.Keyboard", return_value=mock_instance)
        return mock_instance

    def test_press_with_keycode(self, mock_keyboard):
        """Test pressing a key using KeyCode enum."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.press(KeyCode.A)
        mock_keyboard.press.assert_called_once_with(KeyCode.A)

    def test_press_with_string(self, mock_keyboard, mocker):
        """Test pressing a key using a string identifier."""
        kb_output = InputtinoKeyboardOutput()

        kb_output.press("b")
        mock_keyboard.press.assert_called_once_with(KeyCode.B)

    def test_press_with_int(self, mock_keyboard):
        """Test pressing a key using an integer key code."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.press(65)  # ASCII code for 'A'
        mock_keyboard.press.assert_called_once_with(KeyCode(65))

    def test_press_multiple_keys(self, mock_keyboard):
        """Test pressing multiple keys at once."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.press(KeyCode.CTRL, KeyCode.C)

        assert mock_keyboard.press.call_count == 2
        mock_keyboard.press.assert_any_call(KeyCode.CTRL)
        mock_keyboard.press.assert_any_call(KeyCode.C)

    def test_release_with_keycode(self, mock_keyboard):
        """Test releasing a key using KeyCode enum."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.release(KeyCode.A)
        mock_keyboard.release.assert_called_once_with(KeyCode.A)

    def test_release_with_string(self, mock_keyboard, mocker):
        """Test releasing a key using a string identifier."""
        kb_output = InputtinoKeyboardOutput()

        kb_output.release("b")
        mock_keyboard.release.assert_called_once_with(KeyCode.B)

    def test_release_with_int(self, mock_keyboard):
        """Test releasing a key using an integer key code."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.release(65)  # ASCII code for 'A'
        mock_keyboard.release.assert_called_once_with(KeyCode(65))

    def test_release_multiple_keys(self, mock_keyboard):
        """Test releasing multiple keys at once."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.release(KeyCode.CTRL, KeyCode.C)

        assert mock_keyboard.release.call_count == 2
        mock_keyboard.release.assert_any_call(KeyCode.CTRL)
        mock_keyboard.release.assert_any_call(KeyCode.C)
