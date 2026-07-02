from enum import Enum
from source.WhisperConfig import WhisperConfig

class ModelType(Enum):
    WHISPER_LARGE_V3 = WhisperConfig(model_path="C:\\Users\\nish0\\pet_projects\\audio_convert_tier_2\\whisper_models\\large-v3",language="uk",device="cpu")
    # OLLAMA_QWEN_27B = "qwen3.6:27b"
    # OLLAMA_QWEN_14b = "qwen2.5-coder:14b"