from .output import Key, KeyboardOutput

__all__ = ["Key", "KeyboardOutput"]

try:
    from .output.inputtino import InputtinoKeyboardOutput

    __all__.extend(["InputtinoKeyboardOutput"])
except ModuleNotFoundError:
    pass
