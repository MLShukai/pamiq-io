"""Tests for the WindowsKeyboardOutput class."""

from tests.helpers import skip_if_platform_is_not_windows

skip_if_platform_is_not_windows()

import pydirectinput
import pytest
from pytest_mock import MockerFixture

from pamiq_io.keyboard.output.windows import Key, WindowsKeyboardOutput


class TestWindowsKeyboardOutput:
    """Tests for the WindowsKeyboardOutput class."""

    @pytest.fixture
    def mock_directinput(self, mocker: MockerFixture):
        """Create a mock for the pydirectinput module."""
        return mocker.patch("pamiq_io.keyboard.output.windows.pydirectinput")

    @pytest.mark.parametrize(
        "key, expected",
        [
            (Key.A, "a"),
            (Key.CTRL, "ctrl"),
            (Key.LEFT_SHIFT, "shiftleft"),
            (Key.ENTER, "enter"),
            (Key.F1, "f1"),
            (Key.ESC, "esc"),
            (Key.SPACE, "space"),
            (Key.SEMICOLON, ";"),
            (Key.LEFT_SUPER, "winleft"),
        ],
    )
    def test_to_directinput_key(self, key, expected):
        """Test converting Key enum to pydirectinput key strings."""
        assert WindowsKeyboardOutput.to_directinput_key(key) == expected

    @pytest.mark.parametrize("key", Key)
    def test_to_directinput_key_in_key_mapping(self, key):
        """Test converting Key enum to pydirectinput key strings."""
        assert (
            WindowsKeyboardOutput.to_directinput_key(key)
            in pydirectinput.KEYBOARD_MAPPING
        )

    def test_press(self, mock_directinput):
        """Test pressing a key."""
        kb_output = WindowsKeyboardOutput()
        kb_output.press(Key.A)
        mock_directinput.keyDown.assert_called_once_with("a")

    def test_press_multiple_keys(self, mock_directinput):
        """Test pressing multiple keys at once."""
        kb_output = WindowsKeyboardOutput()
        kb_output.press(Key.CTRL, Key.C)

        assert mock_directinput.keyDown.call_count == 2
        mock_directinput.keyDown.assert_any_call("ctrl")
        mock_directinput.keyDown.assert_any_call("c")

    def test_release(self, mock_directinput):
        """Test releasing a key."""
        kb_output = WindowsKeyboardOutput()
        kb_output.release(Key.A)
        mock_directinput.keyUp.assert_called_once_with("a")

    def test_release_multiple_keys(self, mock_directinput):
        """Test releasing multiple keys at once."""
        kb_output = WindowsKeyboardOutput()
        kb_output.release(Key.CTRL, Key.C)

        assert mock_directinput.keyUp.call_count == 2
        mock_directinput.keyUp.assert_any_call("ctrl")
        mock_directinput.keyUp.assert_any_call("c")
