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
        "--fps",
        type=float,
        default=100.0,
        help="Frames per second for velocity updates (default: 100.0)",
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

    # Initialize the mouse output with specified fps
    mouse = InputtinoMouseOutput(fps=args.fps)

    # Parameters
    radius = args.radius
    duration = args.duration

    # Angular velocity (radians per second)
    angular_velocity = 2 * math.pi / duration

    try:
        # Start time for the animation
        start_time = time.perf_counter()
        running = True

        while running:
            # Calculate elapsed time
            current_time = time.perf_counter()
            elapsed = current_time - start_time

            # Check if we've completed the circle
            if elapsed >= duration:
                running = False
                break

            # Calculate current angle
            angle = angular_velocity * elapsed

            # Calculate velocity vector that is tangent to the circle
            # For a circle, the tangent at angle θ is (-sin θ, cos θ)
            vx = -math.sin(angle) * (radius * angular_velocity)
            vy = math.cos(angle) * (radius * angular_velocity)

            # Update mouse velocity
            mouse.move(vx, vy)

            # Log progress (at 10% intervals)
            progress = int(elapsed / duration * 100)
            if progress % 10 == 0:
                logger.info(f"Progress: {progress}%")

            # Small sleep to avoid CPU usage
            time.sleep(0.01)

        # Stop the mouse movement
        mouse.move(0, 0)
        logger.info("Circle drawing completed")

    except KeyboardInterrupt:
        # Stop the mouse movement
        mouse.move(0, 0)
        logger.info("Demo interrupted by user")


if __name__ == "__main__":
    main()
