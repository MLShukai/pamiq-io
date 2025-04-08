#!/usr/bin/env python
"""Demo script for SoundcardAudioOutput.

This script demonstrates playing a 440Hz sine wave for 5 seconds using
SoundcardAudioOutput.
"""

import argparse
import logging

import numpy as np

from pamiq_io.audio import SoundcardAudioOutput


def setup_logging() -> None:
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int,
    channels: int,
) -> np.ndarray:
    """Generate a sine wave with specified parameters.

    Args:
        frequency: Frequency of the sine wave in Hz.
        duration: Duration of the audio in seconds.
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels (1 for mono, 2 for stereo).

    Returns:
        A numpy array containing the sine wave data with shape (frames, channels)
        and values normalized between -1.0 and 1.0.
    """
    # Generate time points
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create sine wave
    sine_wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # If stereo or more channels, duplicate the sine wave across all channels
    if channels > 1:
        sine_wave = np.tile(sine_wave.reshape(-1, 1), (1, channels))
    else:
        sine_wave = sine_wave.reshape(-1, 1)

    return sine_wave


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Play a sine wave using audio output")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Output device ID (default: system default)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=440.0,
        help="Sine wave frequency in Hz (default: 440.0)",
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
        help="Playback duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Block size for audio processing (default: 1024)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Amplitude of the sine wave (0.0-1.0) (default: 0.5)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio output devices and exit",
    )
    return parser.parse_args()


def main() -> None:
    """Run the demo."""
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    # List devices if requested
    if args.list_devices:
        from pamiq_io.audio.output.soundcard import show_all_output_devices

        show_all_output_devices()
        return

    logger.info(f"Initializing audio output (device: {args.device or 'default'})")
    logger.info(f"Sample rate: {args.sample_rate} Hz, Channels: {args.channels}")
    logger.info(f"Sine wave: {args.frequency} Hz, Duration: {args.duration} seconds")
    logger.info(f"Amplitude: {args.amplitude}")

    # Initialize the audio output
    output_device = SoundcardAudioOutput(
        sample_rate=args.sample_rate,
        device_id=args.device,
        block_size=args.block_size,
        channels=args.channels,
    )

    # Generate sine wave
    logger.info("Generating sine wave...")
    audio_data = generate_sine_wave(
        frequency=args.frequency,
        duration=args.duration,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )

    # Scale by amplitude
    audio_data *= args.amplitude

    # Play the audio
    logger.info(f"Playing {args.duration} seconds of {args.frequency} Hz sine wave...")
    output_device.write(audio_data)

    logger.info("Playback completed!")


if __name__ == "__main__":
    main()
