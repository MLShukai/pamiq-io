#!/usr/bin/env python
"""Demo script for InputtinoKeyboardOutput.

This script demonstrates basic keyboard input simulation by pressing and
releasing W, A, S, D keys sequentially with a 1-second delay.
"""

import logging
import time

from pamiq_io.keyboard import InputtinoKeyboardOutput, Key


def setup_logging() -> None:
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """Run the keyboard input simulation demo."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting keyboard input simulation demo")

    # Initialize the keyboard output
    keyboard = InputtinoKeyboardOutput()

    # Define the sequence of keys to press
    keys = [Key.W, Key.A, Key.S, Key.D]

    try:
        # Press each key in sequence with a 1-second delay
        for key in keys:
            logger.info(f"Pressing key: {key}")
            keyboard.press(key)
            time.sleep(1.0)  # Wait 1 second.

            logger.info(f"Releasing key: {key}")
            keyboard.release(key)

        logger.info("Keyboard input simulation completed")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")


if __name__ == "__main__":
    main()
