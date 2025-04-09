"""Keyboard output module for simulating keyboard inputs."""

from inputtino import Keyboard, KeyCode

__all__ = ["InputtinoKeyboardOutput", "KeyCode"]

type Key = int | str | KeyCode


class InputtinoKeyboardOutput:
    """A high-level interface for simulating keyboard inputs.

    This class wraps the inputtino Keyboard class to provide a more
    convenient interface for simulating keyboard inputs. It supports
    pressing and releasing multiple keys at once and accepts keys in
    various formats (KeyCode enum, string, or integer).

    Examples:
        >>> from keyboard_output import KeyboardOutput, KeyCode
        >>> kb = KeyboardOutput()
        >>> kb.press(KeyCode.CTRL, "a")  # Press Ctrl+A
        >>> kb.release(KeyCode.CTRL, "a")  # Release Ctrl+A
    """

    def __init__(self) -> None:
        """Initialize the KeyboardOutput with a virtual keyboard.

        Creates a new instance of inputtino.Keyboard to handle the
        actual keyboard simulation.
        """
        self._keyboard = Keyboard()

    def press(self, *keys: Key) -> None:
        """Press one or more keys simultaneously.

        Args:
            *keys: Variable number of keys to press. Each key can be a
                KeyCode enum value, a string (which will be converted using
                KeyCode.from_str), or an integer key code.
        """
        for k in keys:
            if isinstance(k, str):
                k = KeyCode.from_str(k)
            self._keyboard.press(KeyCode(k))

    def release(self, *keys: Key) -> None:
        """Release one or more keys.

        Args:
            *keys: Variable number of keys to release. Each key can be a
                KeyCode enum value, a string (which will be converted using
                KeyCode.from_str), or an integer key code.
        """
        for k in keys:
            if isinstance(k, str):
                k = KeyCode.from_str(k)
            self._keyboard.release(KeyCode(k))
