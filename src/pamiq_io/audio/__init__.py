from .input import AudioInput
from .output import AudioOutput

__all__ = ["AudioInput", "AudioOutput"]

try:
    from .input.soundcard import SoundcardAudioInput
    from .output.soundcard import SoundcardAudioOutput

    __all__.extend(["SoundcardAudioInput", "SoundcardAudioOutput"])

except ModuleNotFoundError:
    pass
