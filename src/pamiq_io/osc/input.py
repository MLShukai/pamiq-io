"""This module provides OSC input functionality for pamiq-io."""

import logging
import threading
from typing import Any, Protocol

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


class MessageHandler(Protocol):
    """Protocol defining the handler function signature for OSC messages."""

    def __call__(self, address: str, /, *args: Any, **kwds: Any) -> None: ...


class OscInput:
    """OSC input implementation for receiving OSC messages.

    This class allows receiving OSC (Open Sound Control) messages over UDP
    and handles them using registered handler functions.

    Examples:
        >>> def handler(*args):
        ...     print(f"Received message: {args}")
        >>> osc_input = OscInput(host="127.0.0.1", port=9001)
        >>> osc_input.add_handler("/test", handler)
        >>> osc_input.start(blocking=False)  # Start in non-blocking mode
        >>> # ... do other things while receiving OSC messages ...
        >>> osc_input.stop()  # Stop the server when done
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9001) -> None:
        """Initializes an instance of OscInput.

        Args:
            host: The IP address to bind the OSC server to.
            port: The port number to listen on for OSC messages.
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create the dispatcher for routing OSC messages
        self._dispatcher = Dispatcher()

        # Server instance (will be created when start() is called)
        self._server = None
        self._server_thread = None

    def add_handler(self, address: str, handler: MessageHandler) -> None:
        """Add a handler function for a specific OSC address pattern.

        Args:
            address: The OSC address pattern to match (e.g., "/test").
            handler: The handler function to be called when a matching message is received.
                     The handler function should accept any number of arguments
                     extracted from the OSC message.

        Examples:
            >>> def my_handler(address: str, *args):
            ...     print(f"Received message: {args} from {address}")
            >>> osc_input.add_handler("/test", my_handler)
        """

        self._dispatcher.map(address, handler)  # pyright: ignore[reportUnknownMemberType]

    def start(self, blocking: bool = False) -> None:
        """Start the OSC server to begin receiving messages.

        Args:
            blocking: If True, this method will block the current thread until stop() is called.
                     If False, the server will run in a separate thread.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._server_thread is not None:
            raise RuntimeError("OSC input server is already running")

        self._server = ThreadingOSCUDPServer((self.host, self.port), self._dispatcher)
        if blocking:
            self.logger.info(
                f"Starting OSC input server at {self.host}:{self.port} (blocking mode)"
            )

            try:
                self._server.serve_forever()
            finally:
                self._server = None
        else:
            self._server_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self.logger.info(
                f"Starting OSC input server at {self.host}:{self.port} (non-blocking mode)"
            )
            self._server_thread.start()

    def stop(self) -> None:
        """Stop the OSC server.

        This method stops the server if it is running. If the server was started
        in non-blocking mode, this will also wait for the server thread to terminate.

        Raises:
            RuntimeError: If the server is not running.
        """
        if self._server is None:
            raise RuntimeError("OSC input server is not running")

        self.logger.info("Stopping OSC input server")
        self._server.shutdown()

        # For non-blocking mode, wait for the thread to finish
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)

        self._server = None
        self._server_thread = None

    def __del__(self) -> None:
        """Cleanup method to properly close the OSC server when the object is
        destroyed."""
        if hasattr(self, "_server") and self._server is not None:
            self.stop()
