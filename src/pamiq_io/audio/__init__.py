from .input import AudioInput
from .output import AudioOutput
from .utils import AudioFrame

__all__ = ["AudioInput", "AudioOutput", "AudioFrame"]

try:
    from .input.soundcard import SoundcardAudioInput
    from .output.soundcard import SoundcardAudioOutput

    __all__.extend(["SoundcardAudioInput", "SoundcardAudioOutput"])

except ModuleNotFoundError:
    pass
