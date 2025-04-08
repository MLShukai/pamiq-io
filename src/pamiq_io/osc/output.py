"""This module provides OSC output functionality for pamiq-io."""

import logging
from collections.abc import Mapping

from pythonosc.udp_client import SimpleUDPClient

# Type aliases for OSC message values
type MessageValue = str | bytes | int | float | bool
type MessageValueOrValues = MessageValue | list[MessageValue]


class OscOutput:
    """OSC (Open Sound Control) output client for sending messages over UDP.

    This class provides a simple interface for sending OSC messages to a target
    host and port. It wraps the python-osc library's SimpleUDPClient for easy
    integration with pamiq-io.

    Examples:
        >>> osc_out = OscOutput(host="127.0.0.1", port=9001)
        >>> osc_out.send("/test/address", 42.0)
        >>> osc_out.send_messages({
        ...     "/slider/1": 0.5,
        ...     "/button/toggle": True,
        ...     "/multi/value": [1, 2, 3]
        ... })
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9001) -> None:
        """Initialize an OSC output client.

        Args:
            host: The target host address (IP or hostname).
                Defaults to "127.0.0.1" (localhost).
            port: The target port number. Defaults to 9001.
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create the OSC client
        self._client = SimpleUDPClient(host, port)
        self.logger.debug(f"Initialized OSC output to {host}:{port}")

    def send(self, address: str, value: MessageValueOrValues) -> None:
        """Send a single OSC message.

        Args:
            address: The OSC address pattern (e.g., "/test/address").
            value: The value or list of values to send.
        """
        self._client.send_message(address, value)  # pyright: ignore[reportUnknownMemberType]
        self.logger.debug(f"Sent OSC message to {address}: {value}")

    def send_messages(self, messages: Mapping[str, MessageValueOrValues]) -> None:
        """Send multiple OSC messages.

        Args:
            messages: A mapping of OSC address patterns to their corresponding values.
        """
        for addr, value in messages.items():
            self.send(addr, value)
