import sys

from .output import MouseButton, MouseOutput

__all__ = ["MouseOutput", "MouseButton"]

if sys.platform == "linux":
    try:
        from .output.inputtino import InputtinoMouseOutput

        __all__.extend(["InputtinoMouseOutput"])
    except ModuleNotFoundError:
        pass

if sys.platform == "win32":
    try:
        from .output.windows import WindowsMouseOutput

        __all__.extend(["WindowsMouseOutput"])
    except ModuleNotFoundError:
        pass
