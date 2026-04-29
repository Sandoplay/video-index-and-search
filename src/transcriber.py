import os
import wave
from pathlib import Path
from typing import Any, Dict, Optional

class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "turbo",
        backend: str = "faster",
        beam_size: int = 3,
        batch_size: int = 16,
        batched: bool = True,
        vad_filter: bool = False,
        device: str = "cuda",
        compute_type: str = "float16",
        cuda_dll_dir: str = os.path.join("bin", "llama-cpp"),
    ):
        if backend not in {"openai", "faster"}:
            raise ValueError(f"Unsupported Whisper backend: {backend}")

        self.model_size = model_size
        self.backend = backend
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.batched = batched
        self.vad_filter = vad_filter
        self.device = device
        self.compute_type = compute_type
        self.cuda_dll_dir = cuda_dll_dir
        self._model = None
        self._dll_handle = None

    def __del__(self):
        if self._dll_handle is not None:
            self._dll_handle.close()

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_size,
            "beam_size": self.beam_size,
            "batched": self.batched if self.backend == "faster" else None,
            "batch_size": self.batch_size if self.backend == "faster" else None,
            "vad_filter": self.vad_filter if self.backend == "faster" else None,
            "device": self.device if self.backend == "faster" else None,
            "compute_type": self.compute_type if self.backend == "faster" else None,
        }

    def _add_windows_dll_directory(self):
        if os.name != "nt" or not self.cuda_dll_dir:
            return

        dll_dir = Path(self.cuda_dll_dir).resolve()
        if not dll_dir.exists():
            raise RuntimeError(
                f"CUDA DLL directory not found: {dll_dir}. "
                "Set --cuda_dll_dir to a directory containing CUDA 12 DLLs for faster-whisper."
            )

        missing = [
            name for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")
            if not (dll_dir / name).exists()
        ]
        if self.device == "cuda" and missing:
            raise RuntimeError(
                f"Missing CUDA DLLs for faster-whisper in {dll_dir}: {', '.join(missing)}. "
                "Add CUDA 12 cuBLAS/runtime DLLs or use --whisper_device cpu."
            )

        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory") and self._dll_handle is None:
            self._dll_handle = os.add_dll_directory(str(dll_dir))

    def _audio_duration(self, audio_path: str) -> float:
        try:
            with wave.open(audio_path, "rb") as wav:
                rate = wav.getframerate()
                if rate > 0:
                    return wav.getnframes() / float(rate)
        except wave.Error:
            pass
        return 0.0

    def _full_audio_clips(self, audio_path: str, chunk_seconds: float = 30.0) -> list[dict[str, float]]:
        duration = self._audio_duration(audio_path)
        if duration <= 0:
            return [{"start": 0.0, "end": chunk_seconds}]

        clips = []
        start = 0.0
        while start < duration:
            end = min(start + chunk_seconds, duration)
            clips.append({"start": start, "end": end})
            start = end
        return clips

    @property
    def model(self):
        if self._model is None:
            os.makedirs("models", exist_ok=True)

            if self.backend == "openai":
                try:
                    import whisper
                except ImportError as exc:
                    raise RuntimeError("openai-whisper is not installed. Install it with: pip install openai-whisper") from exc

                print(f"Loading OpenAI Whisper model: {self.model_size} into models/ directory...")
                self._model = whisper.load_model(self.model_size, download_root="models")
            else:
                self._add_windows_dll_directory()
                try:
                    from faster_whisper import BatchedInferencePipeline, WhisperModel
                except ImportError as exc:
                    raise RuntimeError("faster-whisper is not installed. Install it with: pip install faster-whisper") from exc

                print(
                    f"Loading faster-whisper model: {self.model_size} "
                    f"(device={self.device}, compute_type={self.compute_type})..."
                )
                model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=os.path.join("models", "faster-whisper"),
                )
                self._model = BatchedInferencePipeline(model=model) if self.batched else model
        return self._model

    def transcribe(self, audio_path: str) -> str:
        """Transcribes the audio file and returns the text."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.backend == "openai":
            result = self.model.transcribe(
                audio_path,
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            return result["text"].strip()

        options: Dict[str, Any] = {
            "beam_size": self.beam_size,
            "condition_on_previous_text": False,
            "vad_filter": self.vad_filter,
            "word_timestamps": False,
            "without_timestamps": True,
        }
        if self.batched:
            options["batch_size"] = self.batch_size
            if not self.vad_filter:
                options["clip_timestamps"] = self._full_audio_clips(audio_path)

        segments, _info = self.model.transcribe(audio_path, **options)
        return "".join(segment.text for segment in segments).strip()
