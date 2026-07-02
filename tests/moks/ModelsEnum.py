from enum import Enum
from source.WhisperConfig import WhisperConfig
from pathlib import Path


MODEL_PATH:Path = Path(__file__).parent / "whisper_models/large-v3"

class ModelType(Enum):
    WHISPER_LARGE_V3 = WhisperConfig(model_path=MODEL_PATH,language="uk",device="cpu")
    # OLLAMA_QWEN_27B = "qwen3.6:27b"
    # OLLAMA_QWEN_14b = "qwen2.5-coder:14b"