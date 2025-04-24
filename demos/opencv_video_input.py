#!/usr/bin/env python
"""Demo script for OpenCVVideoInput.

This script demonstrates capturing a single frame from a camera using
OpenCVVideoInput and saving it as a PNG file using PIL.
"""

import argparse
import logging
from pathlib import Path

from PIL import Image

from pamiq_io.video import OpenCVVideoInput


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
    parser = argparse.ArgumentParser(
        description="Capture a frame from camera and save as PNG"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of captured frame (default: use camera's default)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of captured frame (default: use camera's default)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS of capture (default: use camera's default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="captured_frame.png",
        help="Output file path (default: captured_frame.png)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the demo.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing camera (index: {args.camera})")
    if args.width is not None or args.height is not None or args.fps is not None:
        resolution_info = []
        if args.width is not None:
            resolution_info.append(f"width: {args.width}")
        if args.height is not None:
            resolution_info.append(f"height: {args.height}")
        if args.fps is not None:
            resolution_info.append(f"FPS: {args.fps}")
        logger.info(f"Requested camera parameters: {', '.join(resolution_info)}")
    else:
        logger.info("Using camera's default parameters")

    # Initialize the video input
    input_device = OpenCVVideoInput(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    # Log actual camera parameters (might differ from requested)
    logger.info(
        f"Actual resolution: {input_device.width}x{input_device.height}, FPS: {input_device.fps}"
    )

    # Capture a single frame
    logger.info("Capturing frame...")
    frame = input_device.read()

    # Convert NumPy array to PIL Image and save
    logger.info(f"Saving frame to {output_path}")
    im = Image.fromarray(frame)
    im.save(output_path, format="PNG")

    logger.info("Frame captured and saved successfully!")


if __name__ == "__main__":
    main()
