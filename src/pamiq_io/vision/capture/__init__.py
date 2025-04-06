"""Video capture module for computer vision tasks."""

from .base import VideoCapture
from .opencv import OpenCVVideoCapture

__all__ = ["VideoCapture", "OpenCVVideoCapture"]
