import os
import shutil
import subprocess
import re
import math
from pathlib import Path
from typing import List, Optional
from .models import VideoMetadata, FrameMetadata

class MediaProcessor:
    def __init__(self, output_root: str = "data/processed"):
        self.output_root = output_root
        self.ffmpeg_bin = os.path.join("bin", "ffmpeg.exe")
        self.ffprobe_bin = os.path.join("bin", "ffprobe.exe")
        
        # Ensure binaries exist
        if not os.path.exists(self.ffmpeg_bin):
            # Fallback to system path
            self.ffmpeg_bin = "ffmpeg"
        if not os.path.exists(self.ffprobe_bin):
            self.ffprobe_bin = "ffprobe"

    def _get_video_duration(self, video_path: str) -> float:
        """Gets video duration in seconds."""
        try:
            cmd = [
                self.ffprobe_bin, 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration: {e}")
            return 0.0

    def prepare_video_metadata(self, video_path: str) -> VideoMetadata:
        """Creates the output directory and base metadata structure."""
        video_name = Path(video_path).stem
        # Sanitize name
        video_name = re.sub(r'[^\w\-]', '_', video_name)
        
        base_dir = os.path.join(self.output_root, video_name)
        os.makedirs(base_dir, exist_ok=True)
        
        duration = self._get_video_duration(video_path)
        
        return VideoMetadata(
            video_path=video_path,
            base_dir=base_dir,
            duration=duration
        )

    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extracts audio to WAV (16kHz mono) for Whisper."""
        try:
            # We assume metadata is already prepared or we just infer the path
            video_name = Path(video_path).stem
            video_name = re.sub(r'[^\w\-]', '_', video_name)
            output_path = os.path.join(self.output_root, video_name, "audio.wav")
            
            if os.path.exists(output_path):
                print(f"Audio already exists: {output_path}")
                return output_path

            # -ac 1: mono, -ar 16000: 16kHz
            cmd = [
                self.ffmpeg_bin, "-y",
                "-i", video_path,
                "-ac", "1",
                "-ar", "16000",
                output_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Audio extraction failed: {e}")
            return None

    def extract_frames(self, video_path: str, max_frames: int = 40) -> List[FrameMetadata]:
        """
        Adaptively extracts frames based on content and duration.
        - max_frames: Maximum limit of frames to output (downsamples if more are found).
        """
        video_name = Path(video_path).stem
        video_name = re.sub(r'[^\w\-]', '_', video_name)
        frames_dir = os.path.join(self.output_root, video_name, "frames")
        
        # Clean previous frames to avoid mixing
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        duration = self._get_video_duration(video_path)
        print(f"Video duration: {duration:.2f}s")

        # --- Adaptive Logic ---
        # 1. Determine a "min_interval" to prevent bursting.
        #    For short videos (<30s), allow frames every 0.5s.
        #    For long videos (>2m), allow frames every 2.0s.
        if duration < 60:
            min_interval = 0.5
        elif duration < 300:
            min_interval = 2.0
        else:
            min_interval = 5.0

        # 2. FFmpeg Filter
        # select='gt(scene,0.15)' -> Select frames with >15% change
        # OR 'not(mod(n,K))' -> Force a frame every K frames (fallback)
        # But simply using scene + min_interval is safer.
        # We will extract slightly MORE than needed, then downsample in Python.
        
        # Filter: "Select if Scene Change > 15% AND time since last selected > min_interval"
        # Note: FFmpeg's select filter state is tricky. 
        # Easier approach: Use a relatively sensitive scene threshold, but output vsync to keep timestamps.
        
        # We will use a standard sensitive scene detection, then filter by count.
        output_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
        
        # Scene threshold 0.15 is a good balance for "significant motion".
        # We also force a frame every 'force_interval' seconds to catch static slides.
        force_interval = max(duration / (max_frames * 1.5), 5.0) # Ensure we have at least candidates
        force_fps = 1 / force_interval
        
        select_filter = f"gt(scene,0.15)"
        
        cmd = [
            self.ffmpeg_bin, "-y",
            "-i", video_path,
            "-vf", f"select='{select_filter}',vsync=vfr", # Variable frame rate
            "-q:v", "2", # High quality JPEG
            output_pattern
        ]
        
        # If video is extremely static, the scene filter might return 0 frames.
        # We handle this by checking result count.
        
        print(f"Running adaptive frame extraction...")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 3. Collect extracted frames
        frames = []
        for file in sorted(os.listdir(frames_dir)):
            if file.endswith(".jpg"):
                # We need timestamps. FFmpeg doesn't write timestamp to filename easily without complex configuration.
                # However, with vsync=vfr, it's hard to guess.
                # Alternative: Use ffprobe to read timestamps of extracted JPEGs? No, JPEGs don't hold video timestamps.
                # BETTER: Enable debug output in ffmpeg to capture timestamps? Complicated.
                
                # RETRY with timestamp in filename using frame properties?
                # Actually, simpler method for "Adaptivity" with timestamps:
                # Use standard FPS sampling but HIGH fps, then scene detect in Python? Too slow.
                pass

        # Since getting exact timestamps from `vsync=vfr` output filenames is hard (frame_001 doesn't tell us time),
        # We will switch to a robust "fps" filter strategy which is deterministic, 
        # OR we rely on the fact that we can re-probe the creation time? No.
        
        # ROBUST APPROACH:
        # Use `showinfo` to get timestamps? 
        # Let's use the strategy where we extract at a fixed HIGH frequency (e.g. 1fps) 
        # AND scene changes, then prune.
        
        # ACTUALLY: Let's use `fps=1` (or calculated) -> deterministic.
        # If we want 20-50 frames.
        # Target FPS = max_frames / duration.
        # e.g. 50 frames / 10s = 5 fps.
        # e.g. 50 frames / 100s = 0.5 fps.
        
        target_raw_fps = max_frames / duration if duration > 0 else 1
        # Clamp FPS to reasonable bounds
        target_raw_fps = min(max(target_raw_fps, 0.2), 5.0) 
        
        print(f"Targeting raw extraction at ~{target_raw_fps:.2f} fps based on duration.")
        
        # Re-run extraction with fps filter (reliable timestamps via math)
        # filename frame_0001.jpg -> timestamp ~ 1/fps * (i-1)
        shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        
        cmd = [
            self.ffmpeg_bin, "-y",
            "-i", video_path,
            "-vf", f"fps={target_raw_fps},scale=-2:768",
            "-q:v", "3",
            output_pattern
        ]
        
        # Спроба 1: Звичайний запуск
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Якщо сталася помилка (наприклад, Invalid color space)
        if result.returncode != 0:
            print(f"  -> Warning: Extraction failed. Attempting to fix metadata for {video_name}...")
            
            # Створюємо тимчасовий "вилікуваний" файл
            temp_fixed = os.path.join(self.output_root, video_name, "temp_fixed.mp4")
            
            fix_cmd = [
                self.ffmpeg_bin, "-y",
                "-i", video_path,
                "-c", "copy",
                "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
                temp_fixed
            ]
            
            # Лікуємо файл
            fix_result = subprocess.run(fix_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if fix_result.returncode == 0:
                print("  -> File fixed! Retrying extraction...")
                # Оновлюємо шлях до відео в команді на вилікуваний файл
                cmd[cmd.index(video_path)] = temp_fixed
                
                # Спроба 2: Запуск з вилікуваним файлом
                retry_result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if retry_result.returncode != 0:
                     print(f"  -> Critical Error: Even fixed file failed to extract frames: {retry_result.stderr.decode('utf-8')}")
                     
                # Видаляємо тимчасовий файл, щоб не засмічувати диск
                if os.path.exists(temp_fixed):
                    os.remove(temp_fixed)
            else:
                print(f"  -> Critical Error: Failed to fix video metadata: {fix_result.stderr.decode('utf-8')}")
        
        # Collect
        collected_frames = []
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        
        for i, file in enumerate(files):
            # Estimate timestamp
            timestamp = i / target_raw_fps
            collected_frames.append(FrameMetadata(
                path=os.path.join(frames_dir, file),
                timestamp=timestamp,
                index=i
            ))
            
        # 4. Downsampling if we still exceeded max_frames significantly
        if len(collected_frames) > max_frames:
            print(f"Downsampling from {len(collected_frames)} to {max_frames} frames.")
            step = len(collected_frames) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            final_frames = [collected_frames[i] for i in indices]
            
            # Delete unused
            used_paths = {f.path for f in final_frames}
            for f in collected_frames:
                if f.path not in used_paths:
                    os.remove(f.path)
            
            frames = final_frames
        else:
            frames = collected_frames

        print(f"Extracted {len(frames)} frames (Adaptive).")
        return frames
