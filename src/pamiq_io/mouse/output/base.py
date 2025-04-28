from abc import ABC, abstractmethod
from enum import StrEnum


class MouseButton(StrEnum):
    """Enumeration of mouse buttons that can be pressed or released."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    SIDE = "side"
    EXTRA = "extra"


class MouseOutput(ABC):
    """Abstract base class defining interface for mouse output operations.

    This class provides an interface for simulating mouse movements and
    button actions. Implementations of this class can be used to control
    the mouse cursor position and simulate button presses/releases.
    """

    @abstractmethod
    def move(self, vx: float, vy: float) -> None:
        """Set the mouse cursor movement velocity.

        Args:
            vx: Horizontal velocity in pixels per second (positive is right, negative is left)
            vy: Vertical velocity in pixels per second (positive is down, negative is up)
        """
        ...

    @abstractmethod
    def press(self, button: MouseButton) -> None:
        """Press a mouse button.

        Args:
            button: The button to press.
        """
        ...

    @abstractmethod
    def release(self, button: MouseButton) -> None:
        """Release a mouse button.

        Args:
            button: The button to release.
        """
        ...
