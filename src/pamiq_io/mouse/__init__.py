from .output import MouseButton, MouseOutput

__all__ = ["MouseOutput", "MouseButton"]

try:
    import inputtino as _

    from .output.inputtino import InputtinoMouseOutput

    __all__.extend(["InputtinoMouseOutput"])
except ModuleNotFoundError:
    pass
