"""Computer vision related utilities for pamiq-io."""

from .input import VideoInput

__all__ = ["VideoInput"]

try:
    from .input.opencv import OpenCVVideoInput

    __all__.extend(["OpenCVVideoInput"])

except ModuleNotFoundError:
    pass
