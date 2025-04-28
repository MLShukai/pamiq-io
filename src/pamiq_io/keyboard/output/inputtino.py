from typing import override

import inputtino as _core

from .base import Key, KeyboardOutput

KEY_CODE_FAIL_BACKS = {
    Key.LEFT_SUPER: _core.KeyCode.LEFT_WIN,
    Key.RIGHT_SUPER: _core.KeyCode.RIGHT_WIN,
}


class InputtinoKeyboardOutput(KeyboardOutput):
    """A high-level interface for simulating keyboard inputs.

    This class wraps the inputtino Keyboard class to provide a more
    convenient interface for simulating keyboard inputs.
    """

    def __init__(self) -> None:
        """Initialize the KeyboardOutput with a virtual keyboard.

        Creates a new instance of inputtino.Keyboard to handle the
        actual keyboard simulation.
        """
        super().__init__()
        self._keyboard = _core.Keyboard()

    @staticmethod
    def to_inputtino_key_code(key: Key) -> _core.KeyCode:
        try:
            return _core.KeyCode[key.name]
        except KeyError:
            pass
        try:
            return KEY_CODE_FAIL_BACKS[key]
        except KeyError:
            raise KeyError(f"Key {key} can not be converted to Inputtino KeyCode.")

    @override
    def press(self, *keys: Key) -> None:
        """Press one or more keys simultaneously.

        Args:
            *keys: Variable number of keys to press.
        """
        for k in keys:
            self._keyboard.press(self.to_inputtino_key_code(k))

    @override
    def release(self, *keys: Key) -> None:
        """Release one or more keys.

        Args:
            *keys: Variable number of keys to release.
        """
        for k in keys:
            self._keyboard.release(self.to_inputtino_key_code(k))
