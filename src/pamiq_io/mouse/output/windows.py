"""Mouse output module for simulating mouse inputs with Windows on Windows."""

import sys

if sys.platform == "win32":
    import threading
    import time
    from typing import override

    import pydirectinput

    from .base import MouseButton, MouseOutput

    class WindowsMouseOutput(MouseOutput):
        """Mouse output implementation for Windows using Windows.

        This class provides a high-level interface for simulating mouse movements
        and button actions using pydirectinput. It uses a background thread
        to create smooth mouse movements based on specified velocities.

        Examples:
            >>> from pamiq_io.mouse import WindowsMouseOutput
            >>> mouse = WindowsMouseOutput()
            >>> mouse.move(100, 50)  # Move mouse at 100px/s right, 50px/s down
            >>> mouse.press(MouseButton.LEFT)  # Press left mouse button
            >>> mouse.release(MouseButton.LEFT)  # Release left mouse button
        """

        @override
        def __init__(self, fps: float = 60.0) -> None:
            """Initialize the WindowsMouseOutput.

            Creates a mouse controller using pydirectinput and starts a background
            thread for smooth movement.

            Args:
                fps: The target frame rate for mouse movement updates
            """
            super().__init__()
            self._interval = 1 / fps
            pydirectinput.PAUSE = self._interval

            # Current movement speed (pixels per second)
            self._dx = 0.0
            self._dy = 0.0

            # Accumulated fractional movement
            self._accumulated_dx = 0.0
            self._accumulated_dy = 0.0

            # Thread control
            self._running = True
            self._velocity_lock = threading.Lock()
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
            with self._velocity_lock:
                self._dx = vx * self._interval
                self._dy = vy * self._interval

        @staticmethod
        def convert_to_directinput_button(button: MouseButton) -> str:
            """Convert a MouseButton enum value to a Windows button string.

            Args:
                button: The button to convert.

            Returns:
                The corresponding Windows button string.
            """
            # pydirectinput uses lowercase button names as strings
            match button:
                case MouseButton.SIDE:
                    return pydirectinput.PRIMARY
                case MouseButton.EXTRA:
                    return pydirectinput.SECONDARY
                case _:
                    return button.value

        @override
        def press(self, button: MouseButton) -> None:
            """Press a mouse button.

            Args:
                button: The button to press.
            """
            with self._mouse_lock:
                pydirectinput.mouseDown(
                    button=self.convert_to_directinput_button(button)
                )

        @override
        def release(self, button: MouseButton) -> None:
            """Release a mouse button.

            Args:
                button: The button to release.
            """
            with self._mouse_lock:
                pydirectinput.mouseUp(button=self.convert_to_directinput_button(button))

        def _update_loop(self) -> None:
            """Background thread update loop for mouse movement."""
            last_time = time.perf_counter()
            while self._running:
                with self._velocity_lock:
                    # Calculate movement for this frame
                    self._accumulated_dx += self._dx
                    self._accumulated_dy += self._dy

                    # Extract integer part for movement
                    move_dx, self._accumulated_dx = divmod(self._accumulated_dx, 1)
                    move_dy, self._accumulated_dy = divmod(self._accumulated_dy, 1)

                # Apply mouse movement
                move_dx, move_dy = int(move_dx), int(move_dy)
                if move_dx != 0 or move_dy != 0:
                    with self._mouse_lock:
                        pydirectinput.moveRel(
                            xOffset=move_dx, yOffset=move_dy, relative=True
                        )
                # Maintain frame rate
                if (
                    sleep_time := self._interval - (time.perf_counter() - last_time)
                ) > 0:
                    time.sleep(sleep_time)
                last_time = time.perf_counter()

        def __del__(self) -> None:
            """Clean up resources when the object is being destroyed.

            Stops the background thread safely.
            """
            self.move(0, 0)
            self._running = False
            if hasattr(self, "_thread") and self._thread.is_alive():
                self._thread.join(timeout=1.0)
