# ↔️ pamiq-io

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Document Style](https://img.shields.io/badge/%20docstyle-google-3666d6.svg)](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)

**pamiq-io** is a versatile I/O library for Python, providing easy access to audio, video, and input device capabilities for interactive applications, simulations, and AI projects, made for P-AMI\<Q>.

## ✨ Features

- 🎤 Audio input/output via SoundCard
- 📹 Video input via OpenCV
- 🎛️ OSC (Open Sound Control) communication
- ⌨️ Keyboard simulation
- 🖱️ Mouse simulation

## 🔧 Requirements

- Python 3.12+
- Linux (due to Inputtino and other dependencies)
- OBS Studio (for video capture)
- PulseAudio (for audio)

## 📦 Installation

### Using pip

**For keyboard and mouse output, you must first install [inputtino](https://github.com/games-on-whales/inputtino/tree/stable/bindings/python).**

```bash
# Install the base package
pip install pamiq-io

# Install with optional dependencies as needed
pip install pamiq-io[inputtino]    # For keyboard and mouse output
pip install pamiq-io[opencv]       # For video input
pip install pamiq-io[osc]          # For OSC communication
pip install pamiq-io[soundcard]    # For audio input/output

# For running demo scripts
sudo apt install libsndfile1
pip install pamiq-io[demo]
```

### Development installation

```bash
# Install build dependencies
sudo apt install git cmake build-essential pkg-config libevdev-dev libsndfile1

# Clone and setup
git clone https://github.com/MLShukai/pamiq-io.git
cd pamiq-io
make venv     # Sets up virtual environment with all dependencies
```

## 🧰 Command-Line Tools

pamiq-io includes several helpful command-line tools:

```bash
# List available video input devices
pamiq-io-show-opencv-available-input-devices

# List available audio input devices
pamiq-io-show-soundcard-available-input-devices

# List available audio output devices
pamiq-io-show-soundcard-available-output-devices
```

## 🛠️ Setup

### OBS Virtual Camera

1. Install OBS Studio following the official installation instructions:

   - Visit [https://obsproject.com](https://obsproject.com)
   - Follow the installation guide for Linux (typically using a PPA for Ubuntu-based distributions)

2. If the virtual camera functionality is not available after installing OBS, you may need to install v4l2loopback:

   ```bash
   sudo apt install v4l2loopback-dkms
   sudo modprobe v4l2loopback
   ```

3. In OBS, start the virtual camera (Tools → Start Virtual Camera)

4. (Optional) To find the virtual camera device, you can install v4l-utils:

   ```bash
   sudo apt install v4l-utils
   v4l2-ctl --list-devices | grep -A 1 'OBS Virtual Camera' | grep -oP '\t\K/dev.*'
   ```

## 🐳 Docker

A Docker configuration is provided for easy development and deployment.

### Basic usage:

```dockerfile
# Build a basic image with required dependencies
FROM ubuntu:latest

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git cmake build-essential pkg-config libevdev-dev clang \
    libopencv-dev \
    libsndfile1 \
    pulseaudio \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pamiq-io with desired optional dependencies
RUN pip install "git+https://github.com/games-on-whales/inputtino.git#subdirectory=bindings/python&branch=stable" && \
    pip install pamiq-io[inputtino,opencv,osc,soundcard,demo]

# For development, you may want to check our devcontainer configuration:
# https://github.com/MLShukai/pamiq-io/blob/main/.devcontainer/Dockerfile
```

When running the container, you need privileged access for hardware devices:

```bash
docker run --privileged -it your-pamiq-image
```

> \[!IMPORTANT\]
> ⚠️ **Note**: The `--privileged` flag is required for hardware access to input devices.

### PulseAudio in Docker

To use audio inside Docker, you need to set up PulseAudio properly:

```bash
docker run --privileged -it \
    -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
    -v $HOME/.config/pulse/cookie:/root/.config/pulse/cookie \
    -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
    -e PULSE_COOKIE=/root/.config/pulse/cookie \
    your-pamiq-image
```

## 📚 Usage

### Video Input

```python
# If OpenCV is installed:
from pamiq_io.video import OpenCVVideoInput
from pamiq_io.video.input.opencv import show_video_devices

# List available video devices
show_video_devices()

# Capture from camera using default parameters
video_input = OpenCVVideoInput(camera=0)
frame = video_input.read()

# Capture with specific resolution
video_input = OpenCVVideoInput(camera=0, width=640, height=480, fps=30.0)
frame = video_input.read()

# Capture with mixed parameters (use default width, but specify height)
video_input = OpenCVVideoInput(camera=0, width=None, height=720, fps=None)
frame = video_input.read()
```

### Audio Input/Output

```python
# If SoundCard is installed:
from pamiq_io.audio import SoundcardAudioInput, SoundcardAudioOutput
from pamiq_io.audio.input.soundcard import show_all_input_devices
from pamiq_io.audio.output.soundcard import show_all_output_devices

# List available devices
show_all_input_devices()
show_all_output_devices()

# Capture audio
audio_input = SoundcardAudioInput(sample_rate=44100, channels=2)
audio_data = audio_input.read(frame_size=1024)

# Play audio
import numpy as np
sample_rate = 44100
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
sine_wave = np.sin(2 * np.pi * 440 * t).reshape(-1, 1).astype(np.float32)  # 440Hz

audio_output = SoundcardAudioOutput(sample_rate=sample_rate, channels=1)
audio_output.write(sine_wave)
```

### OSC Communication

```python
# If python-osc is installed:
from pamiq_io.osc import OscOutput, OscInput

# Send OSC messages
osc_output = OscOutput(host="127.0.0.1", port=9001)
osc_output.send("/test/address", 42)

# Receive OSC messages
def handler(addr, value):
    print(f"Received {value} from {addr}")

osc_input = OscInput(host="127.0.0.1", port=9001)
osc_input.add_handler("/test/address", handler)
osc_input.start(blocking=False)
```

### Keyboard Simulation

#### Inputtino

```python
# If inputtino is installed:
from pamiq_io.keyboard import Key, InputtinoKeyboardOutput

# Using the InputtinoKeyboardOutput implementation
keyboard = InputtinoKeyboardOutput()
keyboard.press(Key.CTRL, Key.C)  # Press Ctrl+C
keyboard.release(Key.CTRL, Key.C)  # Release Ctrl+C
```

### Mouse Simulation

#### Inputtino

```python
# If inputtino is installed:
from pamiq_io.mouse import MouseButton, InputtinoMouseOutput

# Using the InputtinoMouseOutput implementation
mouse = InputtinoMouseOutput(fps=100)
mouse.move(100, 50)  # Move 100 pixels/sec right, 50 pixels/sec down
mouse.press(MouseButton.LEFT)
mouse.release(MouseButton.LEFT)
```

## 🧪 Demo Scripts

The repo includes several demo scripts to help you get started:

```bash
# Audio demos (requires pamiq-io[soundcard,demo])
python demos/soundcard_audio_input.py --list-devices
python demos/soundcard_audio_output.py --frequency 440 --duration 3

# Video demos (requires pamiq-io[opencv,demo])
python demos/opencv_video_input.py --camera 0 --output frame.png

# OSC demos (requires pamiq-io[osc])
python demos/osc_io.py

# Input simulation demos (requires pamiq-io[inputtino])
python demos/inputtino_keyboard_output.py
python demos/inputtino_mouse_output.py --radius 100 --duration 5
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request
