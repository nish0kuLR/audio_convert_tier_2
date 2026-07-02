import os
import glob
import ctypes
import importlib.util

def _find_dll_dir(package_name: str) -> str | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None or not spec.submodule_search_locations:
        return None
    base = list(spec.submodule_search_locations)[0]
    for sub in ("bin", "lib"):
        candidate = os.path.join(base, sub)
        if os.path.isdir(candidate):
            return candidate
    return base

def preload_nvidia_dlls() -> None:
    packages = ["nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"]
    dll_paths: list[str] = []

    for pkg in packages:
        d = _find_dll_dir(pkg)
        if d is None:
            print(f"Warning: '{pkg}' not found (skipping)")
            continue
        os.add_dll_directory(d)  
        dll_paths.extend(glob.glob(os.path.join(d, "*.dll")))

    remaining = dll_paths
    while remaining:
        still_failing = []
        for path in remaining:
            try:
                ctypes.WinDLL(path)
            except OSError:
                still_failing.append(path)
        if len(still_failing) == len(remaining):
            break  # no progress this pass, stop
        remaining = still_failing

    if remaining:
        names = [os.path.basename(p) for p in remaining]
        print(f"Warning: could not preload: {names}")

preload_nvidia_dlls()


from pathlib import Path
import numpy as np

from source.FasterWhisperConverter import FasterWhisperConverter
from source.StreamingConverter import StreamingConverter
from source.audio_sources.MicrophoneSource import MicrophoneSource
from source.WhisperConfig import WhisperConfig

#rewrite MODEL_PATH for installed model
MODEL_PATH:Path = Path(__file__).parent / "whisper_models/large-v3"

def main() -> None:
    model_to_use = WhisperConfig(model_path=MODEL_PATH, language="ru", device="cuda")
    converter = FasterWhisperConverter()
    streaming_converter = StreamingConverter(converter)
    source = MicrophoneSource(block_size=24000)

    print("loading model, this can take a while on cpu")
    warmup_chunk:np.ndarray = np.zeros(16000, dtype=np.float32)
    converter.convert_audio_to_text(warmup_chunk, model_to_use)
    print("model loaded")

    source.open()
    print("listening, ctrl+c to stop")

    try:
        while source.is_active():
            audio_chunk = source.get_chunk()
            live_result = streaming_converter.process_chunk(audio_chunk, model_to_use)
            print(live_result)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()

if __name__ == "__main__":
    main()