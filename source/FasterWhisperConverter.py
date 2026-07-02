from pathlib import Path
from faster_whisper import WhisperModel
import numpy as np

from source.WhisperConfig import WhisperConfig
from interfaces.IAudioConverter import IAudioConverter


class FasterWhisperConverter(IAudioConverter):

    def __init__(self):
        self._cached_model: WhisperModel | None = None
        self._cached_model_signature: WhisperConfig | None = None

    def _load_model(self, model_type: WhisperConfig) -> None:
        model_path: Path = model_type.model_path
        device_to_use: str = model_type.device
        try:
            self._cached_model = WhisperModel(
                model_size_or_path=str(model_path),
                device=device_to_use,
                compute_type="int8" if device_to_use == "cpu" else "float16",
            )
            self._cached_model_signature = model_type
        except Exception as exc:
            raise ValueError(
                f"Failed to load Whisper model from '{model_path}' "
                f"on device '{device_to_use}': {exc}"
            ) from exc

    def convert_audio_to_text(
        self,
        audio_sample: Path | np.ndarray,
        model_to_use: WhisperConfig,
        context: str = "",
    ) -> list:
        if self._cached_model_signature != model_to_use:
            self._load_model(model_type=model_to_use)

        language_to_use: str = model_to_use.language
        audio_input = str(audio_sample) if isinstance(audio_sample, Path) else audio_sample

        segments, _ = self._cached_model.transcribe(
            audio_input,
            language=language_to_use,
            beam_size=5,
            word_timestamps=False,
            condition_on_previous_text=True,
            vad_filter=True,
            task="transcribe",
            initial_prompt=context,
            temperature=0,
        )
        return list(segments)