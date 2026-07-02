from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WhisperConfig:
    model_path: Path
    language: str
    device: str