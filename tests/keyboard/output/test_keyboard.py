"""Tests for the keyboard_output module."""

import inputtino as _core
import pytest

from pamiq_io.keyboard.output.inputtino import InputtinoKeyboardOutput, Key


class TestInputtinoKeyboardOutput:
    """Tests for the InputtinoKeyboardOutput class."""

    @pytest.fixture
    def mock_keyboard(self, mocker):
        """Create a mock for the Keyboard class."""
        mock_instance = mocker.MagicMock()
        mocker.patch("inputtino.Keyboard", return_value=mock_instance)
        return mock_instance

    @pytest.mark.parametrize("key", Key)
    def test_inputtino_key_code_conversion(self, key):
        assert isinstance(
            InputtinoKeyboardOutput.to_inputtino_key_code(key), _core.KeyCode
        )

    def test_press(self, mock_keyboard):
        """Test pressing a key using KeyCode enum."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.press(Key.A)
        mock_keyboard.press.assert_called_once_with(_core.KeyCode.A)

    def test_press_multiple_keys(self, mock_keyboard):
        """Test pressing multiple keys at once."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.press(Key.CTRL, Key.C)

        assert mock_keyboard.press.call_count == 2
        mock_keyboard.press.assert_any_call(_core.KeyCode.CTRL)
        mock_keyboard.press.assert_any_call(_core.KeyCode.C)

    def test_release(self, mock_keyboard):
        """Test releasing a key using KeyCode enum."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.release(Key.A)
        mock_keyboard.release.assert_called_once_with(_core.KeyCode.A)

    def test_release_multiple_keys(self, mock_keyboard):
        """Test releasing multiple keys at once."""
        kb_output = InputtinoKeyboardOutput()
        kb_output.release(Key.CTRL, Key.C)

        assert mock_keyboard.release.call_count == 2
        mock_keyboard.release.assert_any_call(_core.KeyCode.CTRL)
        mock_keyboard.release.assert_any_call(_core.KeyCode.C)
