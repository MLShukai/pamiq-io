#!/usr/bin/env python
"""Demo script for OSC Input and Output.

This script demonstrates OSC communication by creating a loopback setup
where messages are sent and received within the same process.
"""

import argparse
import logging
import time
from typing import Any

from pamiq_io.osc import OscInput, OscOutput


def setup_logging() -> None:
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Demo for OSC loopback communication")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=9001, help="Port number (default: 9001)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval between messages in seconds (default: 1.0)",
    )
    return parser.parse_args()


def message_handler(address: str, value: Any) -> None:
    """Handle received OSC messages.

    Args:
        address: The address of OSC message.
        values: The values received in the OSC message.
    """
    logger = logging.getLogger("osc-receiver")
    logger.info(f"Received message: {value} from {address}")


def main() -> None:
    """Run the OSC loopback demo."""
    setup_logging()
    logger = logging.getLogger("osc-demo")
    args = parse_args()

    address = "/pamiq/test"
    logger.info(f"Starting OSC loopback demo on {args.host}:{args.port}")

    # Set up OSC input (receiver)
    osc_in = OscInput(host=args.host, port=args.port)
    osc_in.add_handler(address, message_handler)
    osc_in.start(blocking=False)
    logger.info(f"Listening for OSC messages at {address}")

    # Set up OSC output (sender)
    osc_out = OscOutput(host=args.host, port=args.port)
    logger.info(f"Sender configured to send messages to {address}")

    try:
        # Message counter
        count = 1

        # Main loop
        logger.info("Starting message loop. Press Ctrl+C to stop.")
        while True:
            # Create a test message with increasing counter
            message = f"Test message #{count}"
            logger.info(f"Sending: {message}")

            # Send the message
            osc_out.send(address, message)

            # Increment counter
            count += 1

            # Wait for next iteration
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Stopping OSC loopback demo...")
    finally:
        # Clean up
        logger.info("Cleaning up resources...")
        osc_in.stop()
        logger.info("Demo stopped.")


if __name__ == "__main__":
    main()
