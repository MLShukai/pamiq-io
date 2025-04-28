from .output import MouseButton, MouseOutput

__all__ = ["MouseOutput", "MouseButton"]

try:
    from .output.inputtino import InputtinoMouseOutput

    __all__.extend(["InputtinoMouseOutput"])
except ModuleNotFoundError:
    pass
