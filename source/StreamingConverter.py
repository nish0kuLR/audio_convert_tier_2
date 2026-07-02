import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path

from source.WhisperConfig import WhisperConfig
from interfaces.IAudioConverter import IAudioConverter


class StreamingConverter:

    _SAMPLE_RATE: int = 16_000
    _OVERLAP_SAMPLES: int = _SAMPLE_RATE * 2   
    _MAX_BUFFER_SAMPLES: int = _SAMPLE_RATE * 30  #

    def __init__(self, converter: IAudioConverter):
        self._converter: IAudioConverter = converter
        self._audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._results_buffer: list[list[str]] = []
        self._result: str = ""
        self._prefix_offset: int = 0

    def _insert_new_audio_chunck(self, buffer: np.ndarray, audio_chunk: ArrayLike) -> np.ndarray:
        return np.append(buffer, audio_chunk)

    def _convert_on_insert(
        self,
        model_to_use: WhisperConfig,
        converter: IAudioConverter,
        audio_buffer: np.ndarray | Path,
        context: str = "",
    ) -> list[str]:
        text_segment = converter.convert_audio_to_text(audio_buffer, model_to_use=model_to_use, context=context)
        return [s.text.strip() for s in text_segment]

    def _check_prefix(self, buffer: list[list[str]]) -> list[str] | None:
        if len(buffer) < 2:
            return None

        prev: list[str] = buffer[-2]
        curr: list[str] = buffer[-1]

        confirmed: list[str] = []
        for word_prev, word_curr in zip(prev, curr):
            if word_prev == word_curr:
                confirmed.append(word_prev)
            else:
                break

        return confirmed if confirmed else None

    def _flush_buffers(self) -> None:
        self._audio_buffer = self._audio_buffer[-self._OVERLAP_SAMPLES:]
        self._results_buffer = []
        self._prefix_offset = 0

    def reset(self) -> None:
        """Clear all accumulated state to start a fresh transcription session."""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._results_buffer = []
        self._result = ""
        self._prefix_offset = 0

    def process_chunk(self, audio_chunk: ArrayLike, model_to_use: WhisperConfig, context: str = "") -> str:
        self._audio_buffer = self._insert_new_audio_chunck(self._audio_buffer, audio_chunk)
        text: list[str] = self._convert_on_insert(model_to_use, self._converter, self._audio_buffer, context)
        self._results_buffer.append(text)

        confirmed: list[str] | None = self._check_prefix(self._results_buffer)
        if confirmed:
            new_confirmed: list[str] = confirmed[self._prefix_offset:]
            if new_confirmed:
                self._result = (self._result + " " + " ".join(new_confirmed)).strip()
                self._prefix_offset = len(confirmed)

        unconfirmed_tail: list[str] = text[self._prefix_offset:]
        live_result: str = (self._result + " " + " ".join(unconfirmed_tail)).strip()

        if len(self._results_buffer) > 2:
            self._results_buffer = self._results_buffer[-2:]
        if len(self._audio_buffer) > self._MAX_BUFFER_SAMPLES:
            self._flush_buffers()

        return live_result