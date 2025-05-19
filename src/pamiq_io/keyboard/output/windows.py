"""Keyboard output module for simulating keyboard inputs with Windows on
Windows."""

import sys

if sys.platform == "win32":
    from typing import override

    import pydirectinput

    from .base import Key, KeyboardOutput

    # Mapping from Key enum to pydirectinput key strings
    KEY_MAPPING = {
        # Windows keys
        Key.LEFT_SUPER: "winleft",
        Key.RIGHT_SUPER: "winright",
        # Special Control keys
        Key.LEFT_CONTROL: "ctrlleft",
        Key.RIGHT_CONTROL: "ctrlright",
        # OEM Keys
        Key.SEMICOLON: ";",
        Key.PLUS: "=",  # In keyboard layouts, '+' usually requires Shift + '='
        Key.COMMA: ",",
        Key.MINUS: "-",
        Key.PERIOD: ".",
        Key.SLASH: "/",
        Key.TILDE: "`",  # In keyboard layouts, '~' usually requires Shift + '`'
        Key.OPEN_BRACKET: "[",
        Key.BACKSLASH: "\\",
        Key.CLOSE_BRACKET: "]",
        Key.QUOTE: "'",
    }

    class WindowsKeyboardOutput(KeyboardOutput):
        """Keyboard output implementation for Windows.

        This class provides a high-level interface for simulating keyboard inputs
        using pydirectinput library.

        Examples:
            >>> from pamiq_io.keyboard import WindowsKeyboardOutput
            >>> keyboard = WindowsKeyboardOutput()
            >>> keyboard.press(Key.CTRL, Key.C)  # Press Ctrl+C
            >>> keyboard.release(Key.CTRL, Key.C)  # Release Ctrl+C
        """

        def __init__(self) -> None:
            """Initialize the WindowsKeyboardOutput.

            Creates a keyboard controller using pydirectinput to handle
            the actual keyboard simulation.
            """
            super().__init__()

        @staticmethod
        def to_directinput_key(key: Key) -> str:
            """Convert a Key enum value to a pydirectinput key string.

            Args:
                key: The key to convert.

            Returns:
                The corresponding pydirectinput key string.

            Raises:
                KeyError: If the key cannot be converted to a pydirectinput key.
            """
            # Check if key has a specific mapping
            if key in KEY_MAPPING:
                return KEY_MAPPING[key]

            # Numbers
            if key in Key.number_keys():
                return key.name[len("KEY_") :]

            # Function keys
            if key in Key.function_keys():
                return key.name.lower()

            # Letter keys
            if key in Key.letter_keys():
                return key.name.lower()

            # Shift / Alt keys
            if key in (Key.alt_keys() | Key.shift_keys()):
                return "".join(reversed(key.name.split("_"))).lower()

            return key.name.replace("_", "").lower()

        @override
        def press(self, *keys: Key) -> None:
            """Press one or more keys simultaneously.

            Args:
                *keys: Variable number of keys to press.
            """
            for k in keys:
                pydirectinput.keyDown(self.to_directinput_key(k))

        @override
        def release(self, *keys: Key) -> None:
            """Release one or more keys.

            Args:
                *keys: Variable number of keys to release.
            """
            for k in keys:
                pydirectinput.keyUp(self.to_directinput_key(k))
