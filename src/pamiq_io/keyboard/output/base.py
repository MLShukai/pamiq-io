from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto


class KeyboardOutput(ABC):
    """Abstract base class defining interface for keyboard output operations.

    This class provides an interface for simulating keyboard button
    actions. Implementations of this class can be used to simulate key
    presses and releases.
    """

    @abstractmethod
    def press(self, *keys: Key) -> None:
        """Press one or more keys simultaneously.

        Args:
            *keys: Variable number of keys to press.
        """
        ...

    @abstractmethod
    def release(self, *keys: Key) -> None:
        """Release one or more keys.

        Args:
            *keys: Variable number of keys to release.
        """
        ...


class Key(Enum):
    """Enumeration of keyboard keys that can be pressed or released.

    This enum represents virtual key codes for various keys on a
    standard keyboard.
    """

    # Standard Keys
    BACKSPACE = auto()
    TAB = auto()
    CLEAR = auto()
    ENTER = auto()
    SHIFT = auto()
    CTRL = auto()
    ALT = auto()
    PAUSE = auto()
    CAPS_LOCK = auto()
    ESC = auto()
    SPACE = auto()
    PAGE_UP = auto()
    PAGE_DOWN = auto()
    END = auto()
    HOME = auto()
    LEFT = auto()
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    PRINTSCREEN = auto()
    INSERT = auto()
    DELETE = auto()

    # Numbers
    KEY_0 = auto()
    KEY_1 = auto()
    KEY_2 = auto()
    KEY_3 = auto()
    KEY_4 = auto()
    KEY_5 = auto()
    KEY_6 = auto()
    KEY_7 = auto()
    KEY_8 = auto()
    KEY_9 = auto()

    # Letters
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    Z = auto()

    # Windows Keys
    LEFT_SUPER = auto()
    RIGHT_SUPER = auto()

    # Function Keys
    F1 = auto()
    F2 = auto()
    F3 = auto()
    F4 = auto()
    F5 = auto()
    F6 = auto()
    F7 = auto()
    F8 = auto()
    F9 = auto()
    F10 = auto()
    F11 = auto()
    F12 = auto()

    # Left/Right Keys
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
    LEFT_CONTROL = auto()
    RIGHT_CONTROL = auto()
    LEFT_ALT = auto()
    RIGHT_ALT = auto()

    # OEM Keys
    SEMICOLON = auto()
    PLUS = auto()
    COMMA = auto()
    MINUS = auto()
    PERIOD = auto()
    SLASH = auto()
    TILDE = auto()
    OPEN_BRACKET = auto()
    BACKSLASH = auto()
    CLOSE_BRACKET = auto()
    QUOTE = auto()
