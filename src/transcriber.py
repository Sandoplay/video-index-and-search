import whisper
import os

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"Loading Whisper model: {self.model_size}...")
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        """Transcribes the audio file and returns the text."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = self.model.transcribe(audio_path)
        return result["text"].strip()
