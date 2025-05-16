from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import cache
from typing import Self


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

    @classmethod
    @cache
    def number_keys(cls) -> set[Self]:
        """Get a set of all number keys.

        Returns:
            A set containing all number keys (KEY_0 through KEY_9).
        """
        out: set[Self] = set()
        for key in cls:
            if re.match(r"^KEY_\d$", key.name):
                out.add(key)
        return out

    @classmethod
    @cache
    def function_keys(cls) -> set[Self]:
        """Get a set of all function keys.

        Returns:
            A set containing all function keys (F1 through F12).
        """
        out: set[Self] = set()
        for key in cls:
            if re.match(r"^F\d{1,2}$", key.name):
                out.add(key)
        return out

    @classmethod
    @cache
    def letter_keys(cls) -> set[Self]:
        """Get a set of all letter keys.

        Returns:
            A set containing all letter keys (A through Z).
        """
        out: set[Self] = set()
        for key in cls:
            if len(key.name) == 1 and key.name.isalpha():
                out.add(key)
        return out

    @classmethod
    @cache
    def super_keys(cls) -> set[Self]:
        """Get a set of all super keys.

        Returns:
            A set containing all super keys (LEFT_SUPER, RIGHT_SUPER).
        """
        return {key for key in cls if key.name.endswith("SUPER")}

    @classmethod
    @cache
    def shift_keys(cls) -> set[Self]:
        """Get a set of all shift keys.

        Returns:
            A set containing all shift keys (SHIFT, LEFT_SHIFT, RIGHT_SHIFT).
        """
        return {key for key in cls if key.name.endswith("SHIFT")}

    @classmethod
    @cache
    def alt_keys(cls) -> set[Self]:
        """Get a set of all alt keys.

        Returns:
            A set containing all alt keys (ALT, LEFT_ALT, RIGHT_ALT).
        """
        return {key for key in cls if key.name.endswith("ALT")}

    @classmethod
    @cache
    def ctrl_keys(cls) -> set[Self]:
        """Get a set of all control keys.

        Returns:
            A set containing all control keys (CTRL, LEFT_CONTROL, RIGHT_CONTROL).
        """
        return {
            key for key in cls if key.name.endswith("CONTROL") or key.name == "CTRL"
        }
