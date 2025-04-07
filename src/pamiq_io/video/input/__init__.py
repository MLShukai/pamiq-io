"""Video input module for computer vision tasks."""

from .base import VideoInput
from .opencv import OpenCVVideoInput

__all__ = ["VideoInput", "OpenCVVideoInput"]
