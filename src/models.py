from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class FrameMetadata:
    path: str
    timestamp: float
    index: int = 0
    description: Optional[str] = None

@dataclass
class SegmentMetadata:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    description: str

@dataclass
class VideoMetadata:
    video_path: str
    base_dir: str  # Path to the directory containing frames and audio
    output_key: Optional[str] = None
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    transcription: Optional[str] = None
    summary: Optional[str] = None
    frames: List[FrameMetadata] = field(default_factory=list)
    segments: List[SegmentMetadata] = field(default_factory=list)
    timings: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Optional[str] = None
