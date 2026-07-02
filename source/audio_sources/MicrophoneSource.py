import queue
import numpy as np
import sounddevice as sd

from interfaces.ISource import ISyncSource

class MicrophoneSource(ISyncSource):

    def __init__(self, sample_rate:int = 16000, channels:int = 1, block_size:int = 1024):
        self._sample_rate:int = sample_rate
        self._channels:int = channels
        self._block_size:int = block_size
        self._stream:sd.InputStream|None = None
        self._queue:queue.Queue = queue.Queue()
        self._active:bool = False

    def _on_audio_block(self, indata, frames, time, status) -> None:
        self._queue.put(indata.copy().flatten())

    def open(self) -> None:
        self._stream = sd.InputStream(samplerate=self._sample_rate,
                                       channels=self._channels,
                                       blocksize=self._block_size,
                                       dtype="float32",
                                       callback=self._on_audio_block)
        self._stream.start()
        self._active = True

    def is_active(self) -> bool:
        return self._active

    def get_chunk(self) -> np.ndarray:
        return self._queue.get(block=True)

    def close(self) -> None:
        self._active = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()