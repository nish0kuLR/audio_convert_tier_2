import numpy as np
import pytest
from dataclasses import dataclass

from interfaces.IAudioConverter import IAudioConverter
from source.StreamingConverter import StreamingConverter
from source.WhisperConfig import WhisperConfig
from tests.moks.ModelsEnum import ModelType


@dataclass
class FakeSegment:
    text: str


class FakeConverter(IAudioConverter):
    def __init__(self, responses: list[list[str]]):
        self._responses = iter(responses)

    def _load_model(self, model_type):
        pass

    def convert_audio_to_text(
        self,
        audio_sample,
        model_to_use,
        context: str = "",
    ):
        return [FakeSegment(text) for text in next(self._responses)]


@pytest.fixture
def model_to_use() -> WhisperConfig:
    return ModelType.WHISPER_LARGE_V3.value


@pytest.fixture
def audio_chunk() -> np.ndarray:
    return np.array([0.0], dtype=np.float32)


def test_process_chunk_returns_first_transcription(model_to_use, audio_chunk):
    converter = FakeConverter([
        ["Привіт"],
    ])

    streaming = StreamingConverter(converter)

    result = streaming.process_chunk(audio_chunk, model_to_use)

    assert result == "Привіт"


def test_process_chunk_confirms_common_prefix(model_to_use, audio_chunk):
    converter = FakeConverter([
        ["Привіт"],
        ["Привіт", "друже"],
    ])

    streaming = StreamingConverter(converter)

    streaming.process_chunk(audio_chunk, model_to_use)
    result = streaming.process_chunk(audio_chunk, model_to_use)

    assert result == "Привіт друже"


def test_process_chunk_updates_unconfirmed_tail(model_to_use, audio_chunk):
    converter = FakeConverter([
        ["Я", "люб"],
        ["Я", "люблю"],
    ])

    streaming = StreamingConverter(converter)

    first = streaming.process_chunk(audio_chunk, model_to_use)
    second = streaming.process_chunk(audio_chunk, model_to_use)

    assert first == "Я люб"
    assert second == "Я люблю"


def test_process_chunk_handles_no_common_prefix(model_to_use, audio_chunk):
    converter = FakeConverter([
        ["Привіт"],
        ["Бувай"],
    ])

    streaming = StreamingConverter(converter)

    streaming.process_chunk(audio_chunk, model_to_use)
    result = streaming.process_chunk(audio_chunk, model_to_use)

    assert result == "Бувай"


def test_reset_clears_internal_state(model_to_use, audio_chunk):
    converter = FakeConverter([
        ["Привіт"],
    ])

    streaming = StreamingConverter(converter)

    streaming.process_chunk(audio_chunk, model_to_use)
    streaming.reset()

    assert streaming._audio_buffer.size == 0
    assert streaming._results_buffer == []
    assert streaming._result == ""
    assert streaming._prefix_offset == 0