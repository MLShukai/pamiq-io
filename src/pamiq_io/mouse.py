"""Mouse output module for simulating mouse inputs."""

from typing import Literal

from inputtino import Mouse, MouseButton

__all__ = ["InputtinoMouseOutput", "MouseButton", "Button"]

type ButtonLiteral = Literal["left", "right", "middle", "side", "extra"]
type Button = ButtonLiteral | MouseButton


class InputtinoMouseOutput:
    """Mouse output implementation for simulating mouse inputs.

    This class provides a high-level interface for simulating mouse movements
    and button actions using the inputtino library.

    Examples:
        >>> from pamiq_io.mouse import MouseOutput
        >>> mouse = MouseOutput()
        >>> mouse.move(100, 50)  # Move mouse 100px right, 50px down
        >>> mouse.press("left")  # Press left mouse button
        >>> mouse.release("left")  # Release left mouse button
    """

    def __init__(self) -> None:
        """Initialize the MouseOutput with a virtual mouse.

        Creates a new instance of inputtino.Mouse to handle the actual
        mouse simulation.
        """
        self._mouse = Mouse()

    def move(self, dx: int, dy: int) -> None:
        """Move the mouse cursor by the specified delta.

        Args:
            dx: Horizontal movement in pixels (positive is right, negative is left)
            dy: Vertical movement in pixels (positive is down, negative is up)
        """
        self._mouse.move(dx, dy)

    @staticmethod
    def convert_to_mouse_button(button: Button) -> MouseButton:
        """Convert a button identifier to a MouseButton enum value.

        Args:
            button: The button to convert, either a string literal or a MouseButton enum

        Returns:
            The corresponding MouseButton enum value
        """
        match button:
            case "left":
                return MouseButton.LEFT
            case "right":
                return MouseButton.RIGHT
            case "middle":
                return MouseButton.MIDDLE
            case "side":
                return MouseButton.SIDE
            case "extra":
                return MouseButton.EXTRA
            case _:
                return button

    def press(self, button: Button) -> None:
        """Press a mouse button.

        Args:
            button: The button to press, either a string literal or a MouseButton enum
        """
        self._mouse.press(self.convert_to_mouse_button(button))

    def release(self, button: Button) -> None:
        """Release a mouse button.

        Args:
            button: The button to release, either a string literal or a MouseButton enum
        """
        self._mouse.release(self.convert_to_mouse_button(button))
