import pytest
import numpy as np
from unittest.mock import MagicMock

from source.WhisperConfig import WhisperConfig
from source.StreamingConverter import StreamingConverter
from tests.moks.ModelsEnum import ModelType


@pytest.fixture
def model_to_use() -> WhisperConfig:
    return ModelType.WHISPER_LARGE_V3.value


def _mock_segment(text:str):
    segment = MagicMock()
    segment.text = text
    return segment


@pytest.fixture
def mock_converter():
    converter = MagicMock()
    converter.convert_audio_to_text.side_effect = [
        [_mock_segment("Привет"), _mock_segment(" другалище")],
        [_mock_segment("Привет"), _mock_segment(" другалище"), _mock_segment(" вы")],
    ]
    return converter


@pytest.fixture
def streaming_converter(mock_converter):
    return StreamingConverter(mock_converter)


def test_on_process_chunk_grows_audio_buffer(streaming_converter, model_to_use):
    audio_chunk:np.ndarray = np.array([1, 2, 3], dtype=np.float32)
    streaming_converter.process_chunk(audio_chunk, model_to_use)
    assert np.array_equal(streaming_converter._audio_buffer, audio_chunk)

def test_on_process_chunk_fills_results_buffer(streaming_converter, model_to_use):
    audio_chunk:np.ndarray = np.array([1, 2, 3], dtype=np.float32)
    streaming_converter.process_chunk(audio_chunk, model_to_use)
    assert streaming_converter._results_buffer == [["Привет", "другалище"]]

def test_on_process_chunk_confirms_prefix_after_second_call(streaming_converter, model_to_use):
    audio_chunk:np.ndarray = np.array([1, 2, 3], dtype=np.float32)
    streaming_converter.process_chunk(audio_chunk, model_to_use)
    streaming_converter.process_chunk(audio_chunk, model_to_use)
    assert streaming_converter._result == "Привет другалище"

def test_on_process_chunk_returns_current_transcript(streaming_converter, model_to_use):
    audio_chunk:np.ndarray = np.array([1, 2, 3], dtype=np.float32)
    streaming_converter.process_chunk(audio_chunk, model_to_use)
    result:str = streaming_converter.process_chunk(audio_chunk, model_to_use)
    assert result == "Привет другалище вы"

def test_on_process_chunk_no_confirm_on_first_call(streaming_converter, model_to_use):
    audio_chunk:np.ndarray = np.array([1, 2, 3], dtype=np.float32)
    streaming_converter.process_chunk(audio_chunk, model_to_use)
    assert streaming_converter._result == ""