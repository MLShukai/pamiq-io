"""
Type stubs for the soundcard module.
"""

from typing import Any, Self, override

import numpy as np
from numpy.typing import NDArray

_ffi = ...
_pa = ...

def channel_name_map() -> dict[str, int]:
    """
    Return a dict containing the channel position index for every channel position name string.
    """
    ...

class _PulseAudio:
    """Proxy for communication with Pulseaudio.

    This holds the pulseaudio main loop, and a pulseaudio context.
    Together, these provide the building blocks for interacting with
    pulseaudio.

    This can be used to query the pulseaudio server for sources,
    sinks, and server information, and provides thread-safe access to
    the main pulseaudio functions.

    Any function that would return a `pa_operation *` in pulseaudio
    will block until the operation has finished.
    """
    def __init__(self) -> None: ...
    @property
    def name(self) -> str:
        """Return application name stored in client proplist"""
        ...

    @name.setter
    def name(self, name: str) -> None: ...
    @property
    def source_list(self) -> list[dict[str, Any]]:
        """Return a list of dicts of information about available sources."""
        ...

    def source_info(self, id: str) -> dict[str, Any]:
        """Return a dictionary of information about a specific source."""
        ...

    @property
    def sink_list(self) -> list[dict[str, Any]]:
        """Return a list of dicts of information about available sinks."""
        ...

    def sink_info(self, id: str) -> dict[str, Any]:
        """Return a dictionary of information about a specific sink."""
        ...

    @property
    def server_info(self) -> dict[str, Any]:
        """Return a dictionary of information about the server."""
        ...

    _pa_context_get_source_info_list = ...
    _pa_context_get_source_info_by_name = ...
    _pa_context_get_sink_info_list = ...
    _pa_context_get_sink_info_by_name = ...
    _pa_context_get_client_info = ...
    _pa_context_get_server_info = ...
    _pa_context_get_index = ...
    _pa_context_get_state = ...
    _pa_context_set_name = ...
    _pa_context_drain = ...
    _pa_context_disconnect = ...
    _pa_context_unref = ...
    _pa_context_errno = ...
    _pa_operation_get_state = ...
    _pa_operation_unref = ...
    _pa_stream_get_state = ...
    _pa_sample_spec_valid = ...
    _pa_stream_new = ...
    _pa_stream_get_channel_map = ...
    _pa_stream_drain = ...
    _pa_stream_disconnect = ...
    _pa_stream_unref = ...
    _pa_stream_connect_record = ...
    _pa_stream_readable_size = ...
    _pa_stream_peek = ...
    _pa_stream_drop = ...
    _pa_stream_connect_playback = ...
    _pa_stream_update_timing_info = ...
    _pa_stream_get_latency = ...
    _pa_stream_writable_size = ...
    _pa_stream_write = ...
    _pa_stream_set_read_callback = ...

_pulse = ...

def all_speakers() -> list[_Speaker]:
    """A list of all connected speakers.

    Returns
    -------
    speakers : list(_Speaker)
    """
    ...

def default_speaker() -> _Speaker:
    """The default speaker of the system.

    Returns
    -------
    speaker : _Speaker
    """
    ...

def get_speaker(id: int | str) -> _Speaker:
    """Get a specific speaker by a variety of means.

    Parameters
    ----------
    id : int or str
        can be a backend id string (Windows, Linux) or a device id int (MacOS), a substring of the
        speaker name, or a fuzzy-matched pattern for the speaker name.

    Returns
    -------
    speaker : _Speaker
    """
    ...

def all_microphones(
    include_loopback: bool = False, exclude_monitors: bool = True
) -> list[_Microphone]:
    """A list of all connected microphones.

    By default, this does not include loopbacks (virtual microphones
    that record the output of a speaker).

    Parameters
    ----------
    include_loopback : bool
        allow recording of speaker outputs
    exclude_monitors : bool
        deprecated version of ``include_loopback``

    Returns
    -------
    microphones : list(_Microphone)
    """
    ...

def default_microphone() -> _Microphone:
    """The default microphone of the system.

    Returns
    -------
    microphone : _Microphone
    """
    ...

def get_microphone(
    id: int | str, include_loopback: bool = False, exclude_monitors: bool = True
) -> _Microphone:
    """Get a specific microphone by a variety of means.

    By default, this does not include loopbacks (virtual microphones
    that record the output of a speaker).

    Parameters
    ----------
    id : int or str
        can be a backend id string (Windows, Linux) or a device id int (MacOS), a substring of the
        speaker name, or a fuzzy-matched pattern for the speaker name.
    include_loopback : bool
        allow recording of speaker outputs
    exclude_monitors : bool
        deprecated version of ``include_loopback``

    Returns
    -------
    microphone : _Microphone
    """
    ...

def get_name() -> str:
    """Get application name.

    .. note::
       Currently only works on Linux.

    Returns
    -------
    name : str
    """
    ...

def set_name(name: str) -> None:
    """Set application name.

    .. note::
       Currently only works on Linux.

    Parameters
    ----------
    name :  str
        The application using the soundcard
        will be identified by the OS using this name.
    """
    ...

class _SoundCard:
    def __init__(self, *, id: int | str) -> None: ...
    @property
    def channels(self) -> int | list[int]:
        """int or list(int): Either the number of channels, or a list of
        channel indices. Index -1 is the mono mixture of all channels,
        and subsequent numbers are channel numbers (left, right,
        center, ...)
        """
        ...

    @property
    def id(self) -> Any:
        """object: A backend-dependent unique ID."""
        ...

    @property
    def name(self) -> str:
        """str: The human-readable name of the soundcard."""
        ...

class _Speaker(_SoundCard):
    """A soundcard output. Can be used to play audio.

    Use the :func:`play` method to play one piece of audio, or use the
    :func:`player` method to get a context manager for playing continuous
    audio.

    Multiple calls to :func:`play` play immediately and concurrently,
    while the :func:`player` schedules multiple pieces of audio one
    after another.
    """
    @override
    def __repr__(self) -> str: ...
    def player(
        self,
        samplerate: int,
        channels: int | list[int] = ...,
        blocksize: int | None = ...,
        exclusive_mode: bool = False,
    ) -> _Player:
        """Create Player for playing audio.

        Parameters
        ----------
        samplerate : int
            The desired sampling rate in Hz
        channels : {int, list(int)}, optional
            Play on these channels. For example, ``[0, 3]`` will play
            stereo data on the physical channels one and four.
            Defaults to use all available channels.
            On Linux, channel ``-1`` is the mono mix of all channels.
            On macOS, channel ``-1`` is silence.
        blocksize : int
            Will play this many samples at a time. Choose a lower
            block size for lower latency and more CPU usage.
        exclusive_mode : bool, optional
            Windows only: open sound card in exclusive mode, which
            might be necessary for short block lengths or high
            sample rates or optimal performance. Default is ``False``.

        Returns
        -------
        player : _Player
        """
        ...

    def play(
        self,
        data: NDArray[np.float32],
        samplerate: int,
        channels: int | list[int] = ...,
        blocksize: int = ...,
        exclusive_mode: bool = False,
    ) -> ...:
        """Play some audio data.

        Parameters
        ----------
        data : numpy array
            The audio data to play. Must be a *frames x channels* Numpy array.
        samplerate : int
            The desired sampling rate in Hz
        channels : {int, list(int)}, optional
            Play on these channels. For example, ``[0, 3]`` will play
            stereo data on the physical channels one and four.
            Defaults to use all available channels.
            On Linux, channel ``-1`` is the mono mix of all channels.
            On macOS, channel ``-1`` is silence.
        blocksize : int
            Will play this many samples at a time. Choose a lower
            block size for lower latency and more CPU usage.
        """
        ...

class _Microphone(_SoundCard):
    """A soundcard input. Can be used to record audio.

    Use the :func:`record` method to record one piece of audio, or use
    the :func:`recorder` method to get a context manager for recording
    continuous audio.

    Multiple calls to :func:`record` record immediately and
    concurrently, while the :func:`recorder` schedules multiple pieces
    of audio to be recorded one after another.
    """
    @override
    def __repr__(self) -> str: ...
    @property
    def isloopback(self) -> bool:
        """bool : Whether this microphone is recording a speaker."""
        ...

    def recorder(
        self,
        samplerate: int,
        channels: int | list[int] = ...,
        blocksize: int | None = ...,
        exclusive_mode: bool = ...,
    ) -> _Recorder:
        """Create Recorder for recording audio.

        Parameters
        ----------
        samplerate : int
            The desired sampling rate in Hz
        channels : {int, list(int)}, optional
            Record on these channels. For example, ``[0, 3]`` will record
            stereo data from the physical channels one and four.
            Defaults to use all available channels.
            On Linux, channel ``-1`` is the mono mix of all channels.
            On macOS, channel ``-1`` is silence.
        blocksize : int
            Will record this many samples at a time. Choose a lower
            block size for lower latency and more CPU usage.
        exclusive_mode : bool, optional
            Windows only: open sound card in exclusive mode, which
            might be necessary for short block lengths or high
            sample rates or optimal performance. Default is ``False``.

        Returns
        -------
        recorder : _Recorder
        """
        ...

    def record(
        self,
        numframes: int,
        samplerate: int,
        channels: int | list[int] = ...,
        blocksize: int = ...,
        exclusive_mode: bool = False,
    ) -> NDArray[np.float32]:
        """Record some audio data.

        Parameters
        ----------
        numframes: int
            The number of frames to record.
        samplerate : int
            The desired sampling rate in Hz
        channels : {int, list(int)}, optional
            Record on these channels. For example, ``[0, 3]`` will record
            stereo data from the physical channels one and four.
            Defaults to use all available channels.
            On Linux, channel ``-1`` is the mono mix of all channels.
            On macOS, channel ``-1`` is silence.
        blocksize : int
            Will record this many samples at a time. Choose a lower
            block size for lower latency and more CPU usage.

        Returns
        -------
        data : numpy array
            The recorded audio data. Will be a *frames x channels* Numpy array.
        """
        ...

class _Stream:
    """A context manager for an active audio stream.

    This class is meant to be subclassed. Children must implement the
    `_connect_stream` method which takes a `pa_buffer_attr*` struct,
    and connects an appropriate stream.

    This context manager can only be entered once, and can not be used
    after it is closed.
    """
    def __init__(
        self,
        id: int | str,
        samplerate: int,
        channels: int | list[int],
        blocksize: int = ...,
        name: str = ...,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
    @property
    def latency(self) -> float:
        """float : Latency of the stream in seconds (only available on Linux)"""
        ...

class _Player(_Stream):
    """A context manager for an active output stream.

    Audio playback is available as soon as the context manager is
    entered. Audio data can be played using the :func:`play` method.
    Successive calls to :func:`play` will queue up the audio one piece
    after another. If no audio is queued up, this will play silence.

    This context manager can only be entered once, and can not be used
    after it is closed.
    """
    def play(self, data: NDArray[np.float32]) -> None:
        """Play some audio data.

        Internally, all data is handled as ``float32`` and with the
        appropriate number of channels. For maximum performance,
        provide data as a *frames × channels* float32 numpy array.

        If single-channel or one-dimensional data is given, this data
        will be played on all available channels.

        This function will return *before* all data has been played,
        so that additional data can be provided for gapless playback.
        The amount of buffering can be controlled through the
        blocksize of the player object.

        If data is provided faster than it is played, later pieces
        will be queued up and played one after another.

        Parameters
        ----------
        data : numpy array
            The audio data to play. Must be a *frames x channels* Numpy array.
        """
        ...

class _Recorder(_Stream):
    """A context manager for an active input stream.

    Audio recording is available as soon as the context manager is
    entered. Recorded audio data can be read using the :func:`record`
    method. If no audio data is available, :func:`record` will block until
    the requested amount of audio data has been recorded.

    This context manager can only be entered once, and can not be used
    after it is closed.
    """
    def __init__(self, *args, **kwargs) -> None: ...
    def record(self, numframes: int | None = None) -> NDArray[np.float32]:
        """Record a block of audio data.

        The data will be returned as a *frames × channels* float32
        numpy array. This function will wait until ``numframes``
        frames have been recorded. If numframes is given, it will
        return exactly ``numframes`` frames, and buffer the rest for
        later.

        If ``numframes`` is None, it will return whatever the audio
        backend has available right now. Use this if latency must be
        kept to a minimum, but be aware that block sizes can change at
        the whims of the audio backend.

        If using :func:`record` with ``numframes=None`` after using
        :func:`record` with a required ``numframes``, the last
        buffered frame will be returned along with the new recorded
        block. (If you want to empty the last buffered frame instead,
        use :func:`flush`)

        Parameters
        ----------
        numframes : int, optional
            The number of frames to record.

        Returns
        -------
        data : numpy array
            The recorded audio data. Will be a *frames x channels* Numpy array.
        """
        ...

    def flush(self) -> NDArray[np.float32]:
        """Return the last pending chunk.

        After using the :func:`record` method, this will return the
        last incomplete chunk and delete it.

        Returns
        -------
        data : numpy array
            The recorded audio data. Will be a *frames x channels* Numpy array.
        """
        ...
