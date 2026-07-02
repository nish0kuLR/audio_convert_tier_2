import asyncio
import pytest 
import pytest_asyncio
from pathlib import Path
from faster_whisper import WhisperModel
from source.WhisperConfig import WhisperConfig

from source.FasterWhisperConverter import FasterWhisperConverter
from tests.moks.ModelsEnum import ModelType


TEST_AUDIO_PATH:Path =  Path(__file__).parent / "moks/hello_audio_mok.ogg"


@pytest.fixture
def faster_whisper_converter() -> FasterWhisperConverter:
    return FasterWhisperConverter()

@pytest.fixture
def model_to_use() -> WhisperConfig:
    return ModelType.WHISPER_LARGE_V3.value

def test_on_load_model(faster_whisper_converter, model_to_use):
    model_to_use_signature:Path = model_to_use.model_path
    faster_whisper_converter._load_model(model_to_use)
    assert  faster_whisper_converter._cached_model_signature.model_path == model_to_use_signature

def test_on_convert(faster_whisper_converter, model_to_use):
    proccesing_result = faster_whisper_converter.convert_audio_to_text(TEST_AUDIO_PATH, model_to_use)
    print(proccesing_result)
    assert proccesing_result in ["Привіт", "Привіт!"]