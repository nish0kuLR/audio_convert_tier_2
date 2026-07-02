import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path

from source.WhisperConfig import WhisperConfig
from interfaces.IAudioConverter import IAudioConverter


class StreamingConverter:
    _SAMPLE_RATE: int = 16_000
    _OVERLAP_SAMPLES: int = _SAMPLE_RATE * 2
    _MAX_BUFFER_SAMPLES: int = _SAMPLE_RATE * 30
    _MAX_RESULTS_HISTORY: int = 2

    def __init__(self, converter: IAudioConverter):
        self._converter = converter
        self._audio_buffer = np.array([], dtype=np.float32)
        self._results_buffer: list[list[str]] = []
        self._result = ""
        self._prefix_offset = 0

    def process_chunk(
        self,
        audio_chunk: ArrayLike,
        model_to_use: WhisperConfig,
        context: str = "",
    ) -> str:
        self._append_audio(audio_chunk)

        transcript = self._transcribe(model_to_use, context)

        self._store_transcript(transcript)

        self._commit_confirmed_words()

        live_result = self._build_live_result(transcript)

        self._cleanup()

        return live_result

    def _append_audio(self, audio_chunk: ArrayLike) -> None:
        self._audio_buffer = np.append(self._audio_buffer, audio_chunk)

    def _transcribe(
        self,
        model_to_use: WhisperConfig,
        context: str,
    ) -> list[str]:
        segments = self._converter.convert_audio_to_text(
            self._audio_buffer,
            model_to_use=model_to_use,
            context=context,
        )

        return [segment.text.strip() for segment in segments]

    def _store_transcript(self, transcript: list[str]) -> None:
        self._results_buffer.append(transcript)

    def _commit_confirmed_words(self) -> None:
        confirmed = self._find_confirmed_prefix()

        if confirmed is None:
            return

        new_words = confirmed[self._prefix_offset :]

        if not new_words:
            return

        self._result = (self._result + " " + " ".join(new_words)).strip()
        self._prefix_offset = len(confirmed)

    def _find_confirmed_prefix(self) -> list[str] | None:
        if len(self._results_buffer) < 2:
            return None

        previous = self._results_buffer[-2]
        current = self._results_buffer[-1]

        confirmed = []

        for prev_word, curr_word in zip(previous, current):
            if prev_word != curr_word:
                break
            confirmed.append(prev_word)

        return confirmed or None

    def _build_live_result(self, transcript: list[str]) -> str:
        tail = transcript[self._prefix_offset :]

        return (self._result + " " + " ".join(tail)).strip()

    def _cleanup(self) -> None:
        self._trim_results_history()

        if len(self._audio_buffer) > self._MAX_BUFFER_SAMPLES:
            self._flush_buffers()

    def _trim_results_history(self) -> None:
        self._results_buffer = self._results_buffer[-self._MAX_RESULTS_HISTORY :]

    def _flush_buffers(self) -> None:
        self._audio_buffer = self._audio_buffer[-self._OVERLAP_SAMPLES :]
        self._results_buffer.clear()
        self._prefix_offset = 0

    def reset(self) -> None:
        self._audio_buffer = np.array([], dtype=np.float32)
        self._results_buffer.clear()
        self._result = ""
        self._prefix_offset = 0