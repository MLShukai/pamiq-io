"""Mouse output module for simulating mouse inputs with smooth movement."""

import threading
import time
from typing import override

from inputtino import Mouse, MouseButton as _MouseButton

from .base import MouseButton, MouseOutput


class InputtinoMouseOutput(MouseOutput):
    """Mouse output implementation for simulating mouse inputs with smooth
    movement.

    This class provides a high-level interface for simulating mouse movements
    and button actions using the inputtino library. It uses a background thread
    to create smooth mouse movements based on specified velocities.

    Examples:
        >>> from pamiq_io.mouse import InputtinoMouseOutput
        >>> mouse = InputtinoMouseOutput()
        >>> mouse.move(100, 50)  # Move mouse at 100px/s right, 50px/s down
        >>> mouse.press("left")  # Press left mouse button
        >>> mouse.release("left")  # Release left mouse button
    """

    @override
    def __init__(self, fps: float = 100.0) -> None:
        """Initialize the MouseOutput with a virtual mouse.

        Creates a new instance of inputtino.Mouse to handle the actual
        mouse simulation and starts a background thread for smooth movement.

        Args:
            fps: The target frame rate for mouse movement updates
        """
        super().__init__()
        self._mouse = Mouse()
        self._interval = 1 / fps

        # Current movement speed (pixels per second)
        self._dx = 0
        self._dy = 0

        # Accumulated fractional movement
        self._accumulated_dx = 0.0
        self._accumulated_dy = 0.0

        # Thread control
        self._running = True
        self._velosity_lock = threading.Lock()
        self._mouse_lock = threading.Lock()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    @override
    def move(self, vx: float, vy: float) -> None:
        """Set the mouse cursor movement velocity.

        Sets the speed at which the mouse cursor should move in pixels per second.
        The actual movement is handled by the background thread.

        Args:
            vx: Horizontal velocity in pixels per second (positive is right, negative is left)
            vy: Vertical velocity in pixels per second (positive is down, negative is up)
        """
        with self._velosity_lock:
            self._dx = vx * self._interval
            self._dy = vy * self._interval

    @staticmethod
    def convert_to_mouse_button(button: MouseButton) -> _MouseButton:
        """Convert a button identifier to a MouseButton enum value.

        Args:
            button: The button to convert.

        Returns:
            The corresponding Inputtino MouseButton enum value
        """
        return getattr(_MouseButton, button.upper())

    @override
    def press(self, button: MouseButton) -> None:
        """Press a mouse button.

        Args:
            button: The button to press, either a string literal or a MouseButton enum
        """
        with self._mouse_lock:
            self._mouse.press(self.convert_to_mouse_button(button))

    @override
    def release(self, button: MouseButton) -> None:
        """Release a mouse button.

        Args:
            button: The button to release, either a string literal or a MouseButton enum
        """
        with self._mouse_lock:
            self._mouse.release(self.convert_to_mouse_button(button))

    def _update_loop(self) -> None:
        """Background thread update loop for mouse movement and button
        actions."""
        last_time = time.perf_counter()

        while self._running:
            with self._velosity_lock:
                # Get and Update dx, dy.
                move_dx, self._accumulated_dx = divmod(
                    self._accumulated_dx + self._dx, 1
                )
                move_dy, self._accumulated_dy = divmod(
                    self._accumulated_dy + self._dy, 1
                )

            # Apply mouse movement
            move_dx, move_dy = int(move_dx), int(move_dy)
            if move_dx != 0 or move_dy != 0:
                with self._mouse_lock:
                    self._mouse.move(move_dx, move_dy)

            # Maintain frame rate
            if (sleep_time := self._interval - (time.perf_counter() - last_time)) > 0:
                time.sleep(sleep_time)
            last_time = time.perf_counter()
        self._mouse.move(0, 0)

    def __del__(self) -> None:
        """Clean up resources when the object is being destroyed.

        Stops the background thread safely.
        """
        self.move(0, 0)
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=1.0)
