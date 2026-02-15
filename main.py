import os
# Reduce memory fragmentation and limit large allocations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import json
import torch
import gc
from typing import List
from src.media_processor import MediaProcessor
from src.transcriber import AudioTranscriber
from src.vision_analyzer import VisionAnalyzer
from src.models import VideoMetadata

def cleanup_gpu():
    """Aggressively clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def process_batch(video_paths: List[str], args):
    processor = MediaProcessor(output_root=args.output_dir)
    videos_metadata: List[VideoMetadata] = []

    # --- Stage 1: Media Processing (FFmpeg) ---
    print(f"\n>>> STAGE 1: Media Processing ({len(video_paths)} videos)")
    for path in video_paths:
        print(f"Processing: {os.path.basename(path)}")
        meta = processor.prepare_video_metadata(path)
        meta.audio_path = processor.extract_audio(path)
        meta.frames = processor.extract_frames(path, scene_threshold=args.threshold, max_interval=args.interval)
        videos_metadata.append(meta)
        print(f"  Done. Extracted {len(meta.frames)} frames.")
    
    cleanup_gpu()

    # --- Stage 2: Transcription (Whisper) ---
    print(f"\n>>> STAGE 2: Audio Transcription")
    try:
        transcriber = AudioTranscriber(model_size=args.whisper_model)
        for meta in videos_metadata:
            if meta.audio_path:
                print(f"Transcribing audio for: {os.path.basename(meta.video_path)}")
                meta.transcription = transcriber.transcribe(meta.audio_path)
    finally:
        # Ensure model is deleted even if error occurs
        if 'transcriber' in locals():
            del transcriber
        cleanup_gpu()

    # --- Stage 3: Vision Analysis (Qwen3-VL) ---
    print(f"\n>>> STAGE 3: Visual Analysis")
    try:
        analyzer = VisionAnalyzer(model_id=args.vlm_model)
        for meta in videos_metadata:
            print(f"Analyzing frames for: {os.path.basename(meta.video_path)}")
            if meta.transcription:
                meta.frames = analyzer.analyze_frames(meta.frames, meta.transcription)
            else:
                meta.frames = analyzer.analyze_frames(meta.frames, "No audio context available.")
    finally:
        if 'analyzer' in locals():
            del analyzer
        cleanup_gpu()

    # --- Stage 4: Saving Results ---
    print(f"\n>>> STAGE 4: Saving Results")
    for meta in videos_metadata:
        video_name = os.path.basename(meta.base_dir)
        result_path = os.path.join(meta.base_dir, "analysis_result.json")
        
        result_data = {
            "video_path": meta.video_path,
            "duration": meta.duration,
            "transcription": meta.transcription,
            "analysis": [
                {"timestamp": f.timestamp, "description": f.description, "frame_path": f.path} 
                for f in meta.frames
            ]
        }
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"Results for {video_name} saved to {result_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch Multimodal Video Analysis System")
    parser.add_argument("--input", type=str, required=True, help="Path to video file or directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Root directory for processed data")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="VLM model ID")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--threshold", type=float, default=0.1, help="Scene detection threshold")
    parser.add_argument("--interval", type=float, default=30.0, help="Max interval between frames in seconds")
    
    args = parser.parse_args()

    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    video_paths = []

    if os.path.isfile(args.input):
        video_paths.append(args.input)
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))
    else:
        print(f"Error: Input {args.input} not found.")
        return

    if not video_paths:
        print("No video files found to process.")
        return

    print(f"Found {len(video_paths)} video(s) to process.")
    process_batch(video_paths, args)

if __name__ == "__main__":
    main()
