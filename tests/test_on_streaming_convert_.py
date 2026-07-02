import pytest
import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel
from source.WhisperConfig import WhisperConfig


from source.FasterWhisperConverter import FasterWhisperConverter
from tests.moks.ModelsEnum import ModelType
from interfaces.IAudioConverter import IAudioConverter


from source.StreamingConverter import StreamingConverter

RAW_AUDIO_MOK: Path = Path(__file__).parent / "moks" / "hello.raw"

@pytest.fixture
def model_to_use() -> WhisperConfig:
    return ModelType.WHISPER_LARGE_V3.value

@pytest.fixture
def faster_whisper_converter() -> FasterWhisperConverter:
    return FasterWhisperConverter()

@pytest.fixture
def streaming_converter(faster_whisper_converter):
    return StreamingConverter(faster_whisper_converter)

@pytest.fixture
def audio_buffer():
    return np.array([], dtype=np.float32)

def test_on_chunk_insert(streaming_converter, audio_buffer):
    new_chunk:list[int] = [1]

    result:np.ndarray = streaming_converter._insert_new_audio_chunck(
        audio_buffer,
        new_chunk
    )

    assert np.array_equal(
        result,
        np.array([1], dtype=np.float32)
    )

def test_on_convert_on_insert(streaming_converter, model_to_use, faster_whisper_converter):
    audio_buffer:np.ndarray = np.fromfile(RAW_AUDIO_MOK, dtype=np.float32)

    transcription_result:list[str] = streaming_converter._convert_on_insert(
        model_to_use=model_to_use, 
        converter=faster_whisper_converter, 
        audio_buffer=audio_buffer
    )
    print(transcription_result)
    assert transcription_result is not None

def test_on_prefix_test(streaming_converter):
    buffer:list[list[str]] = [["Привет"," другалище", "в"], ["Привет"," друалище", "ваш", "зять"]]
    confirmed = streaming_converter._check_prefix(buffer=buffer)
    assert confirmed == ["Привет"]