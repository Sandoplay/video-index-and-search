import subprocess
import os
import re
from typing import List
from .models import VideoMetadata, FrameMetadata

class MediaProcessor:
    def __init__(self, output_root: str = "data/processed", bin_dir: str = "bin"):
        self.output_root = output_root
        self.bin_dir = bin_dir
        self.ffmpeg_path = os.path.join(bin_dir, "ffmpeg.exe")
        self.ffprobe_path = os.path.join(bin_dir, "ffprobe.exe")
        os.makedirs(self.output_root, exist_ok=True)

    def _get_video_dir(self, video_path: str) -> str:
        """Creates and returns a specific directory for the video assets."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Sanitize name for folder
        video_name = re.sub(r'[^\w\s-]', '', video_name).strip().replace(' ', '_')
        video_dir = os.path.join(self.output_root, video_name)
        os.makedirs(video_dir, exist_ok=True)
        return video_dir

    def extract_audio(self, video_path: str) -> str:
        """Extracts audio to the video's specific directory."""
        video_dir = self._get_video_dir(video_path)
        audio_path = os.path.join(video_dir, "audio.wav")
        command = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        subprocess.run(command, check=True, capture_output=True)
        return audio_path

    def extract_frames(self, video_path: str, scene_threshold: float = 0.1, max_interval: float = 30.0) -> List[FrameMetadata]:
        """Extracts frames to the video's specific directory."""
        video_dir = self._get_video_dir(video_path)
        # Separate folder for frames inside video_dir for better organization
        frames_dir = os.path.join(video_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        output_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
        filter_str = f"select='gt(scene,{scene_threshold})+gte(t-prev_selected_t,{max_interval})',showinfo"

        command = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-vf", filter_str,
            "-vsync", "vfr", "-q:v", "2",
            output_pattern
        ]
        
        process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
        _, stderr = process.communicate()
        
        frames = []
        frame_index = 1
        for line in stderr.split('\n'):
            if "pts_time:" in line:
                match = re.search(r"pts_time:(\d+\.?\d*)", line)
                if match:
                    timestamp = float(match.group(1))
                    frame_path = os.path.join(frames_dir, f"frame_{frame_index:04d}.jpg")
                    if os.path.exists(frame_path):
                        frames.append(FrameMetadata(path=frame_path, timestamp=timestamp))
                        frame_index += 1
        
        return frames

    def get_video_duration(self, video_path: str) -> float:
        """Returns the duration of the video in seconds."""
        command = [
            self.ffprobe_path, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    
    def prepare_video_metadata(self, video_path: str) -> VideoMetadata:
        """Helper to initialize metadata with correct paths."""
        video_dir = self._get_video_dir(video_path)
        return VideoMetadata(
            video_path=video_path,
            base_dir=video_dir,
            duration=self.get_video_duration(video_path)
        )
