from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FrameMetadata:
    path: str
    timestamp: float
    index: int = 0
    description: Optional[str] = None

@dataclass
class VideoMetadata:
    video_path: str
    base_dir: str  # Path to the directory containing frames and audio
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    transcription: Optional[str] = None
    frames: List[FrameMetadata] = field(default_factory=list)
    analysis_results: Optional[str] = None
