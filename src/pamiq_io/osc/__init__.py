try:
    from .input import OscInput
    from .output import OscOutput

    __all__ = ["OscOutput", "OscInput"]
except ModuleNotFoundError:
    pass
