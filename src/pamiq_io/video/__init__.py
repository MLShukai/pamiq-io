"""Computer vision related utilities for pamiq-io."""

from .input import VideoInput

__all__ = ["VideoInput"]

try:
    import cv2 as _

    from .input.opencv import OpenCVVideoInput

    __all__.extend(["OpenCVVideoInput"])

except ModuleNotFoundError:
    pass
