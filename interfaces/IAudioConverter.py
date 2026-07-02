from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from source.WhisperConfig import WhisperConfig


class IAudioConverter(ABC):

    @abstractmethod
    def _load_model(self, model_type: WhisperConfig) -> None: pass

    @abstractmethod
    def convert_audio_to_text(
        self,
        audio_path: Path | np.ndarray,
        model_to_use: WhisperConfig,
        context: str = "",
    ) -> list: pass