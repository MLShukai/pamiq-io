"""Computer vision related utilities for pamiq-io."""

from .input import VideoInput
from .utils import VideoFrame

__all__ = ["VideoInput", "VideoFrame"]

try:
    from .input.opencv import OpenCVVideoInput

    __all__.extend(["OpenCVVideoInput"])

except ModuleNotFoundError:
    pass
