#!/usr/bin/env python
"""Demo script for InputtinoMouseOutput.

This script demonstrates mouse movement by drawing a circle with the
cursor over 5 seconds.
"""

import argparse
import logging
import math
import time

from pamiq_io.mouse import InputtinoMouseOutput


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
    parser = argparse.ArgumentParser(description="Draw a circle with the mouse cursor")
    parser.add_argument(
        "--radius",
        type=int,
        default=100,
        help="Radius of the circle in pixels (default: 100)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration to complete the circle in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of steps to complete the circle (default: 60)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the mouse movement demo."""
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    logger.info(
        f"Starting mouse circle demo (radius: {args.radius}px, duration: {args.duration}s)"
    )

    # Initialize the mouse output
    mouse = InputtinoMouseOutput()

    # Parameters
    radius = args.radius
    duration = args.duration
    steps = args.steps

    # Time between steps
    step_time = duration / steps

    try:
        # Draw the circle
        for i in range(steps):
            # Calculate current and next angles
            current_angle = 2 * math.pi * i / steps
            next_angle = 2 * math.pi * (i + 1) / steps

            # Calculate coordinates on the circle
            x1 = radius * math.cos(current_angle)
            y1 = radius * math.sin(current_angle)
            x2 = radius * math.cos(next_angle)
            y2 = radius * math.sin(next_angle)

            # Calculate movement needed
            dx = int(x2 - x1)
            dy = int(y2 - y1)

            # Move the mouse
            logger.debug(f"Step {i+1}/{steps}: Moving by dx={dx}, dy={dy}")
            mouse.move(dx, dy)

            # Wait before next step
            time.sleep(step_time)

        logger.info("Circle drawing completed")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")


if __name__ == "__main__":
    main()
