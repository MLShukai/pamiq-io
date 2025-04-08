#!/usr/bin/env python
"""Demo script for SoundcardAudioInput.

This script demonstrates recording audio from a microphone using
SoundcardAudioInput and saving it as a WAV file using soundfile.
"""

import argparse
import logging
from pathlib import Path

import soundfile as sf

from pamiq_io.audio import SoundcardAudioInput


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
        description="Record audio from microphone and save as WAV"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Input device ID (default: system default)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Number of channels (1 for mono, 2 for stereo) (default: 2)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Block size for audio processing (default: 1024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recorded_audio.wav",
        help="Output file path (default: recorded_audio.wav)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
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

    # List devices if requested
    if args.list_devices:
        from pamiq_io.audio.input.soundcard import show_all_input_devices

        show_all_input_devices()
        return

    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing audio input (device: {args.device or 'default'})")
    logger.info(f"Sample rate: {args.sample_rate} Hz, Channels: {args.channels}")
    logger.info(f"Recording duration: {args.duration} seconds")

    # Initialize the audio input
    input_device = SoundcardAudioInput(
        sample_rate=args.sample_rate,
        device_id=args.device,
        block_size=args.block_size,
        channels=args.channels,
    )

    # Calculate number of frames to record based on duration
    total_frames = int(args.sample_rate * args.duration)

    # Record audio
    logger.info(f"Recording {args.duration} seconds of audio...")
    audio_data = input_device.read(frame_size=total_frames)

    # Save the recorded audio
    logger.info(f"Saving audio to {output_path}")
    sf.write(
        file=str(output_path),
        data=audio_data,
        samplerate=int(input_device.sample_rate),
    )

    logger.info("Audio recorded and saved successfully!")


if __name__ == "__main__":
    main()
